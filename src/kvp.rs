use rand::Rng;
use rand::seq::{IndexedRandom, SliceRandom};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::metadata::{METADATA_DELIMITER, MetadataKey};
use crate::types::KvpValue;

/// Represents a single key with one or more value renderings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvpField {
    key: String,
    values: Vec<KvpValue>,
    presence: f32,
}

impl KvpField {
    /// Create a field with exactly one rendering option.
    pub fn one(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self::many(key, [value])
    }

    /// Create a field with multiple rendering options (duplicates ignored, order preserved).
    pub fn many<K, V, I>(key: K, values: I) -> Self
    where
        K: Into<String>,
        I: IntoIterator<Item = V>,
        V: Into<String>,
    {
        let mut seen = HashSet::new();
        let mut collected = Vec::new();
        for value in values.into_iter() {
            let value = value.into();
            if value.is_empty() {
                continue;
            }
            if seen.insert(value.clone()) {
                collected.push(value);
            }
        }
        Self {
            key: key.into(),
            values: collected,
            presence: 1.0,
        }
    }

    /// Override how often this field should appear (0.0=never, 1.0=always).
    pub fn with_presence(mut self, probability: f32) -> Self {
        self.presence = probability.clamp(0.0, 1.0);
        self
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn render<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<String> {
        if self.presence <= 0.0 {
            return None;
        }
        if self.presence < 1.0 && rng.random::<f32>() >= self.presence {
            return None;
        }
        self.values
            .choose(rng)
            .map(|value| format!("{}{}{}", self.key, METADATA_DELIMITER, value))
    }
}

/// Samples key-value "meta" prefixes with optional dropout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvpPrefixSampler {
    dropout: f32,
    variants: Vec<Vec<KvpField>>,
}

impl KvpPrefixSampler {
    /// Create a new sampler that emits prefixes with the provided probability.
    pub fn new(dropout: f32) -> Self {
        Self {
            dropout: dropout.clamp(0.0, 1.0),
            variants: Vec::new(),
        }
    }

    /// Register another variant using simple key-value pairs (single rendering per key).
    pub fn add_variant<K, V, I>(&mut self, fields: I)
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let variant = fields
            .into_iter()
            .map(|(key, value)| KvpField::one(key, value))
            .collect::<Vec<_>>();
        self.add_variant_fields(variant);
    }

    /// Register another variant that may contain multi-valued fields.
    pub fn add_variant_fields<I>(&mut self, fields: I)
    where
        I: IntoIterator<Item = KvpField>,
    {
        let mut variant = Vec::new();
        for field in fields.into_iter() {
            if !field.is_empty() {
                variant.push(field);
            }
        }
        if variant.is_empty() {
            return;
        }
        self.variants.push(variant);
    }

    /// Sample a formatted prefix using the configured dropout rate and variants.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<String> {
        if self.variants.is_empty() || self.dropout <= 0.0 {
            return None;
        }
        if self.dropout < 1.0 && rng.random::<f32>() >= self.dropout {
            return None;
        }
        self.variants
            .choose(rng)
            .and_then(|variant| format_variant(variant, rng))
    }
}

/// Describes how to build a metadata field for a given context (e.g. date).
pub struct MetaFieldSpec<Ctx> {
    key: MetadataKey,
    presence: f32,
    values_builder: fn(&Ctx) -> Vec<KvpValue>,
}

impl<Ctx> MetaFieldSpec<Ctx> {
    pub const fn new(
        key: MetadataKey,
        presence: f32,
        values_builder: fn(&Ctx) -> Vec<KvpValue>,
    ) -> Self {
        Self {
            key,
            presence,
            values_builder,
        }
    }

    fn build(&self, ctx: &Ctx) -> KvpField {
        let values = (self.values_builder)(ctx);
        KvpField::many(self.key.as_str(), values).with_presence(self.presence)
    }
}

/// Holds the ordered list of metadata fields to emit for a source.
pub struct MetaPolicy<Ctx: 'static> {
    fields: &'static [MetaFieldSpec<Ctx>],
}

impl<Ctx: 'static> MetaPolicy<Ctx> {
    pub const fn new(fields: &'static [MetaFieldSpec<Ctx>]) -> Self {
        Self { fields }
    }

    pub fn instantiate(&self, ctx: &Ctx) -> KvpPrefixSampler {
        let built_fields = self
            .fields
            .iter()
            .map(|field| field.build(ctx))
            .collect::<Vec<_>>();
        let mut sampler = KvpPrefixSampler::new(1.0);
        sampler.add_variant_fields(built_fields);
        sampler
    }
}

fn format_variant<R: Rng + ?Sized>(fields: &[KvpField], rng: &mut R) -> Option<String> {
    let mut body = Vec::new();
    for field in fields {
        if let Some(rendered) = field.render(rng) {
            body.push(rendered);
        }
    }
    if body.is_empty() {
        return None;
    }
    if body.len() > 1 {
        body.shuffle(rng);
    }
    Some(format!("meta: {}", body.join(" | ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn respects_dropout_probability() {
        let mut sampler = KvpPrefixSampler::new(1.0);
        sampler.add_variant([("foo", "bar")]);
        let mut rng = StdRng::from_seed([0_u8; 32]);
        assert_eq!(sampler.sample(&mut rng), Some("meta: foo=bar".into()));

        let mut zero_sampler = KvpPrefixSampler::new(0.0);
        zero_sampler.add_variant([("foo", "bar")]);
        assert!(zero_sampler.sample(&mut rng).is_none());
    }

    #[test]
    fn chooses_between_variants() {
        let mut sampler = KvpPrefixSampler::new(1.0);
        sampler.add_variant([("a", "1")]);
        sampler.add_variant([("b", "2")]);
        let mut rng = StdRng::from_seed([1_u8; 32]);
        let first = sampler.sample(&mut rng).unwrap();
        assert!(first == "meta: a=1" || first == "meta: b=2");
    }

    #[test]
    fn ignores_empty_variants() {
        let mut sampler = KvpPrefixSampler::new(1.0);
        sampler.add_variant([] as [(&str, &str); 0]);
        let mut rng = StdRng::from_seed([2_u8; 32]);
        assert!(sampler.sample(&mut rng).is_none());
    }

    #[test]
    fn field_value_options_are_deduped_and_randomized() {
        let field = KvpField::many("date", ["2025-01-01", "Jan 1, 2025", "2025-01-01"]);
        assert_eq!(field.key, "date");
        assert_eq!(field.values, vec!["2025-01-01", "Jan 1, 2025"]);

        let mut rng = StdRng::from_seed([3_u8; 32]);
        let first = field.render(&mut rng).unwrap();
        let second = field.render(&mut rng).unwrap();
        assert!(first == "date=2025-01-01" || first == "date=Jan 1, 2025");
        assert!(second == "date=2025-01-01" || second == "date=Jan 1, 2025");
    }

    #[test]
    fn sampler_handles_multi_value_fields() {
        let mut sampler = KvpPrefixSampler::new(1.0);
        sampler.add_variant_fields([
            KvpField::many("date", ["2025-01-01", "Jan 1, 2025"]),
            KvpField::one("article", "ceo-update"),
        ]);
        let mut rng = StdRng::from_seed([4_u8; 32]);
        let mut outputs = Vec::new();
        for _ in 0..20 {
            if let Some(sample) = sampler.sample(&mut rng) {
                outputs.push(sample);
            }
        }
        outputs.sort();
        outputs.dedup();
        assert!(outputs.len() >= 2);
        assert!(
            outputs
                .iter()
                .any(|value| value.contains("date=2025-01-01")
                    && value.contains("article=ceo-update"))
        );
        assert!(outputs.iter().any(
            |value| value.contains("date=Jan 1, 2025") && value.contains("article=ceo-update")
        ));
    }

    #[test]
    fn sampler_can_shuffle_field_order() {
        let mut sampler = KvpPrefixSampler::new(1.0);
        sampler.add_variant_fields([KvpField::one("alpha", "1"), KvpField::one("beta", "2")]);
        let mut rng = StdRng::from_seed([5_u8; 32]);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..20 {
            if let Some(sample) = sampler.sample(&mut rng) {
                seen.insert(sample);
            }
        }
        assert!(seen.contains("meta: alpha=1 | beta=2"));
        assert!(seen.contains("meta: beta=2 | alpha=1"));
    }

    #[test]
    fn field_presence_controls_dropout() {
        let absent = KvpField::one("foo", "bar").with_presence(0.0);
        let mut rng = StdRng::from_seed([6_u8; 32]);
        assert!(absent.render(&mut rng).is_none());

        let present = KvpField::one("foo", "bar").with_presence(1.0);
        let mut rng2 = StdRng::from_seed([7_u8; 32]);
        assert_eq!(present.render(&mut rng2), Some("foo=bar".into()));
    }
}
