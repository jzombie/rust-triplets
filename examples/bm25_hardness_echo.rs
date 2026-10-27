use std::sync::Arc;

use chrono::Utc;
use triplets_core::data::{DataRecord, QualityScore, RecordSection, SectionRole};
use triplets_core::source::InMemorySource;
use triplets_core::{
    DeterministicSplitStore, RetrievalSamplingConfig, Sampler, SamplerConfig, SplitLabel,
    SplitRatios, TripletRecipe, TripletSampler,
};

fn rec(id: &str, title: &str, body: &str) -> DataRecord {
    DataRecord {
        id: id.to_string(),
        source: "demo".to_string(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore { trust: 1.0 },
        taxonomy: vec![],
        sections: vec![
            RecordSection {
                role: SectionRole::Anchor,
                heading: Some(title.to_string()),
                text: format!("{title}. {body}"),
                sentences: vec![body.to_string()],
                token_count: 0,
                embedding: None,
            },
            RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: body.to_string(),
                sentences: vec![body.to_string()],
                token_count: 0,
                embedding: None,
            },
        ],
        meta_prefix: None,
        label: None,
    }
}

fn sampler_with(cfg_sampling: RetrievalSamplingConfig) -> TripletSampler<DeterministicSplitStore> {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 13).unwrap());
    let cfg = SamplerConfig {
        batch_size: 4,
        split,
        allowed_splits: vec![SplitLabel::Train],
        retrieval_sampling: cfg_sampling.clone(),
        recipes: vec![TripletRecipe {
            name: "demo".into(),
            anchor: triplets_core::config::Selector::Random,
            positive_selector: triplets_core::config::Selector::Random,
            negative_selector: triplets_core::config::Selector::Random,
            negative_strategy: triplets_core::config::NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
            retrieval_sampling: None,
        }],
        ..SamplerConfig::default()
    };
    let s = TripletSampler::new(cfg, store);
    s.register_source(Box::new(InMemorySource::from_records(
        "demo",
        vec![
            rec(
                "doc_005",
                "Marine report",
                "deep sea marine geology survey ocean floor mapping",
            ),
            rec(
                "doc_002",
                "Carbon market update",
                "carbon pricing policy emissions reduction roadmap market design carbon credits",
            ),
            rec(
                "doc_008",
                "Sourdough recipe",
                "flour water salt levain bake crust crumb oven fermentation",
            ),
            rec(
                "doc_001",
                "Energy transition memo",
                "emissions reduction roadmap clean energy transition planning",
            ),
            rec(
                "doc_009",
                "Premier league recap",
                "football soccer goal striker midfield penalty shootout league",
            ),
            rec(
                "doc_004",
                "Climate policy briefing",
                "carbon pricing policy emissions reduction roadmap market design",
            ),
            rec(
                "doc_010",
                "Kubernetes guide",
                "pods nodes containers orchestration cluster deployment scaling",
            ),
            rec(
                "doc_003",
                "Archaeology note",
                "bronze age pottery fragments excavation trench mapping",
            ),
            rec(
                "doc_006",
                "Carbon policy note",
                "carbon pricing policy emissions reduction roadmap",
            ),
            rec(
                "doc_011",
                "Jazz history",
                "bebop saxophone trumpet improvisation swing blue note ensemble",
            ),
            rec(
                "doc_007",
                "Regulatory digest",
                "policy market design regulatory framework compliance",
            ),
            rec(
                "doc_012",
                "Baking croissants",
                "butter layers pastry dough laminate oven flaky breakfast",
            ),
        ],
    )))
    .unwrap();
    s
}

fn token_jaccard(a: &str, b: &str) -> f32 {
    use std::collections::HashSet;
    let sa: HashSet<&str> = a.split_whitespace().collect();
    let sb: HashSet<&str> = b.split_whitespace().collect();
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

fn demo_corpus_texts() -> Vec<(String, String)> {
    vec![
        ("doc_005".to_string(), "Marine report. deep sea marine geology survey ocean floor mapping".to_string()),
        ("doc_002".to_string(), "Carbon market update. carbon pricing policy emissions reduction roadmap market design carbon credits".to_string()),
        ("doc_008".to_string(), "Sourdough recipe. flour water salt levain bake crust crumb oven fermentation".to_string()),
        ("doc_001".to_string(), "Energy transition memo. emissions reduction roadmap clean energy transition planning".to_string()),
        ("doc_009".to_string(), "Premier league recap. football soccer goal striker midfield penalty shootout league".to_string()),
        ("doc_004".to_string(), "Climate policy briefing. carbon pricing policy emissions reduction roadmap market design".to_string()),
        ("doc_010".to_string(), "Kubernetes guide. pods nodes containers orchestration cluster deployment scaling".to_string()),
        ("doc_003".to_string(), "Archaeology note. bronze age pottery fragments excavation trench mapping".to_string()),
        ("doc_006".to_string(), "Carbon policy note. carbon pricing policy emissions reduction roadmap".to_string()),
        ("doc_011".to_string(), "Jazz history. bebop saxophone trumpet improvisation swing blue note ensemble".to_string()),
        ("doc_007".to_string(), "Regulatory digest. policy market design regulatory framework compliance".to_string()),
        ("doc_012".to_string(), "Baking croissants. butter layers pastry dough laminate oven flaky breakfast".to_string()),
    ]
}

fn main() {
    if !cfg!(feature = "bm25-mining") {
        eprintln!(
            "NOTE: running WITHOUT bm25-mining (DefaultBackend uniform). Re-run with --features bm25-mining to compare."
        );
    }
    let corpus = demo_corpus_texts();
    for (label, cfg_sampling) in [
        (
            "RankWindow{1,30} (default)",
            RetrievalSamplingConfig {
                min_rank: 1,
                max_rank: 30,
                search_top_k: 32,
                // Margin off: this example compares rank windows; margin filtering is orthogonal.
                negative_margin: None,
            },
        ),
        (
            "RankWindow{1,10} (extreme hard)",
            RetrievalSamplingConfig {
                min_rank: 1,
                max_rank: 10,
                search_top_k: 32,
                // Margin off: this example compares rank windows; margin filtering is orthogonal.
                negative_margin: None,
            },
        ),
        (
            "RankWindow{0,100} (broad)",
            RetrievalSamplingConfig {
                min_rank: 0,
                max_rank: 100,
                search_top_k: 32,
                // Margin off: this example compares rank windows; margin filtering is orthogonal.
                negative_margin: None,
            },
        ),
    ] {
        println!(
            "
===== {label} ====="
        );
        let s = sampler_with(cfg_sampling);
        let weights = std::collections::HashMap::new();
        match s.next_triplet_batch_with_weights(SplitLabel::Train, &weights) {
            Ok(batch) => {
                for (i, t) in batch.triplets.iter().take(4).enumerate() {
                    let a = t.anchor.text.clone();
                    let n = t.negative.text.clone();
                    let j = token_jaccard(&a, &n);
                    // Uniform baseline: mean Jaccard of anchor vs all other demo docs.
                    let mut sum = 0.0;
                    let mut cnt = 0;
                    for (_, txt) in &corpus {
                        if txt == &a {
                            continue;
                        }
                        let jj = token_jaccard(&a, txt);
                        sum += jj;
                        cnt += 1;
                    }
                    let mean = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
                    let a_head = a.lines().next().unwrap_or("").to_string();
                    let n_head = n.lines().next().unwrap_or("").to_string();
                    println!("[{i}] ANCHOR: {a_head}");
                    println!("    NEG   : {n_head}");
                    println!("    overlap={j:.3} (uniform-mean~{mean:.3})");
                    println!();
                }
            }
            Err(e) => println!("  batch error: {e}"),
        }
    }
}
