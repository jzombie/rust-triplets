use rand::Rng;
use rand::seq::{IndexedRandom, SliceRandom};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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

    /// Returns all metadata keys and all their possible values across every variant.
    ///
    /// This method does **not** involve any RNG, presence rolls, or dropout — it simply
    /// collects every key and every value defined on the sampler. Use the result to
    /// populate [`RecordChunk::kvp_meta`] for downstream inspection/debugging.
    pub fn all_metadata(&self) -> HashMap<String, Vec<String>> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        for variant in &self.variants {
            for field in variant {
                let entry = map.entry(field.key.clone()).or_default();
                for value in &field.values {
                    if !entry.contains(value) {
                        entry.push(value.clone());
                    }
                }
            }
        }
        map
    }
}

/// Describes how to build a metadata field for a given context (e.g. date).
pub struct MetaFieldSpec<Ctx> {
    key: MetadataKey,
    presence: f32,
    values_builder: fn(&Ctx) -> Vec<KvpValue>,
}

impl<Ctx> MetaFieldSpec<Ctx> {
    /// Create a field specification from key, presence probability, and value builder.
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
    /// Create a policy from an ordered static list of field specs.
    pub const fn new(fields: &'static [MetaFieldSpec<Ctx>]) -> Self {
        Self { fields }
    }

    /// Instantiate a `KvpPrefixSampler` for one concrete context value.
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

    #[test]
    fn sampler_and_field_probabilities_are_clamped() {
        let mut always = KvpPrefixSampler::new(2.0);
        always.add_variant([("k", "v")]);
        let mut rng = StdRng::from_seed([8_u8; 32]);
        assert!(always.sample(&mut rng).is_some());

        let mut never = KvpPrefixSampler::new(-1.0);
        never.add_variant([("k", "v")]);
        assert!(never.sample(&mut rng).is_none());

        let field = KvpField::one("a", "b").with_presence(2.0);
        assert_eq!(field.presence, 1.0);
        let field = KvpField::one("a", "b").with_presence(-5.0);
        assert_eq!(field.presence, 0.0);
    }

    #[test]
    fn variant_with_only_absent_fields_returns_none() {
        let mut sampler = KvpPrefixSampler::new(1.0);
        sampler.add_variant_fields([
            KvpField::one("foo", "bar").with_presence(0.0),
            KvpField::many("empty", [""]).with_presence(1.0),
        ]);
        let mut rng = StdRng::from_seed([9_u8; 32]);
        assert!(sampler.sample(&mut rng).is_none());
    }

    #[derive(Clone)]
    struct DemoCtx {
        date: &'static str,
        source: &'static str,
    }

    fn date_values(ctx: &DemoCtx) -> Vec<KvpValue> {
        vec![ctx.date.into()]
    }

    fn source_values(ctx: &DemoCtx) -> Vec<KvpValue> {
        vec![ctx.source.into()]
    }

    const DEMO_DATE_KEY: MetadataKey = MetadataKey::new("date");
    const DEMO_SOURCE_KEY: MetadataKey = MetadataKey::new("source");

    const POLICY_FIELDS: [MetaFieldSpec<DemoCtx>; 2] = [
        MetaFieldSpec::new(DEMO_DATE_KEY, 1.0, date_values),
        MetaFieldSpec::new(DEMO_SOURCE_KEY, 1.0, source_values),
    ];

    #[test]
    fn meta_policy_instantiates_sampler_with_context_values() {
        let policy = MetaPolicy::new(&POLICY_FIELDS);
        let ctx = DemoCtx {
            date: "2026-02-24",
            source: "reports",
        };
        let sampler = policy.instantiate(&ctx);
        let mut rng = StdRng::from_seed([10_u8; 32]);
        let out = sampler.sample(&mut rng).unwrap();
        assert!(out.contains("date=2026-02-24"));
        assert!(out.contains("source=reports"));
    }

    #[test]
    fn kvp_sampler_fractional_dropout_sometimes_suppresses_output() {
        // Covers the `0.0 < dropout < 1.0` branch in KvpPrefixSampler::sample.
        let mut sampler = KvpPrefixSampler::new(0.5);
        sampler.add_variant([("k", "v")]);
        let mut rng = StdRng::from_seed([77_u8; 32]);
        let results: Vec<_> = (0..100).map(|_| sampler.sample(&mut rng)).collect();
        assert!(
            results.iter().any(|r| r.is_none()),
            "dropout=0.5 should suppress some outputs"
        );
        assert!(
            results.iter().any(|r| r.is_some()),
            "dropout=0.5 should pass some outputs"
        );
    }

    #[test]
    fn meta_field_spec_new_is_callable_at_runtime() {
        // Call MetaFieldSpec::new() in a runtime (non-const) context so the
        // constructor body is instrumented by llvm-cov.
        fn values(_: &()) -> Vec<KvpValue> {
            vec!["runtime_val".to_string()]
        }
        let key = MetadataKey::new("runtime_key");
        let spec = MetaFieldSpec::<()>::new(key, 1.0, values);
        let field = spec.build(&());
        let mut rng = StdRng::from_seed([42_u8; 32]);
        assert!(field.render(&mut rng).is_some());
    }

    // ── all_metadata tests ────────────────────────────────────────────────────

    #[test]
    fn all_metadata_empty_when_no_variants() {
        let sampler = KvpPrefixSampler::new(1.0);
        assert!(sampler.all_metadata().is_empty());
    }

    #[test]
    fn all_metadata_collects_all_keys_and_values_regardless_of_dropout() {
        // dropout=0.0 means sample() always returns None, but all_metadata must
        // still expose every declared key and value.
        let mut sampler = KvpPrefixSampler::new(0.0);
        sampler.add_variant_fields([
            KvpField::many("date", ["2025-01-01", "Jan 1, 2025"]),
            KvpField::one("source", "daily-report"),
        ]);

        let meta = sampler.all_metadata();
        assert_eq!(meta.len(), 2);

        let dates = &meta["date"];
        assert_eq!(dates.len(), 2);
        assert!(dates.contains(&"2025-01-01".to_string()));
        assert!(dates.contains(&"Jan 1, 2025".to_string()));

        assert_eq!(meta["source"], vec!["daily-report"]);
    }

    #[test]
    fn all_metadata_collects_keys_across_variants_and_deduplicates_values() {
        // Two variants share the "date" key. all_metadata must merge both variants'
        // values under the same key without duplicates.
        let mut sampler = KvpPrefixSampler::new(1.0);
        sampler.add_variant_fields([
            KvpField::many("date", ["2025-01-01", "Jan 1, 2025"]),
            KvpField::one("source", "variant-a"),
        ]);
        sampler.add_variant_fields([
            KvpField::many("date", ["2025-01-01", "01/01/2025"]), // "2025-01-01" already seen
            KvpField::one("source", "variant-b"),
        ]);

        let meta = sampler.all_metadata();

        // "date" values from both variants, deduped
        let mut dates = meta["date"].clone();
        dates.sort();
        assert_eq!(dates, vec!["01/01/2025", "2025-01-01", "Jan 1, 2025"]);

        // "source" values from both variants
        let mut sources = meta["source"].clone();
        sources.sort();
        assert_eq!(sources, vec!["variant-a", "variant-b"]);
    }

    #[test]
    fn all_metadata_ignores_field_presence_probability() {
        // Fields with presence=0.0 are never sampled, but all_metadata should
        // still include their values.
        let mut sampler = KvpPrefixSampler::new(1.0);
        sampler.add_variant_fields([
            KvpField::one("always", "yes").with_presence(1.0),
            KvpField::one("never", "hidden").with_presence(0.0),
        ]);

        let meta = sampler.all_metadata();
        assert_eq!(meta["always"], vec!["yes"]);
        assert_eq!(meta["never"], vec!["hidden"]);
    }
}
