use triplets_core::SplitLabel;

/// Choose the split most behind its proportional target.
///
/// Returns `None` when every split is exhausted or at its limit.
pub fn next_split_to_fill(
    labels: &[SplitLabel],
    counts: &[u64],
    maxes: &[u64],
    ratios: &[f32],
    exhausted: &[bool],
) -> Option<SplitLabel> {
    debug_assert_eq!(labels.len(), counts.len());
    debug_assert_eq!(counts.len(), maxes.len());
    debug_assert_eq!(counts.len(), ratios.len());
    debug_assert_eq!(counts.len(), exhausted.len());

    let mut best: Option<(SplitLabel, f64)> = None;
    for i in 0..counts.len() {
        if exhausted[i] {
            continue;
        }
        if maxes[i] > 0 && counts[i] >= maxes[i] {
            continue;
        }
        let fullness = if ratios[i] > 0.0 {
            counts[i] as f64 / ratios[i] as f64
        } else {
            f64::MAX
        };
        match best {
            None => best = Some((labels[i], fullness)),
            Some((_, best_fullness)) if fullness < best_fullness => {
                best = Some((labels[i], fullness))
            }
            _ => {}
        }
    }
    best.map(|(label, _)| label)
}

/// Batch-locking split scheduler.
///
/// Wraps the `active_label` / `next_split_to_fill` state machine so it can be
/// tested independently of the main loop.
///
/// **Contract:** `unlock()` must be called after every flush and after marking
/// a split exhausted or at its limit.  Until `unlock()` is called the same
/// split label is returned on every call to `next()`.
pub struct SplitScheduler {
    active_label: Option<SplitLabel>,
}

impl SplitScheduler {
    /// Create a new scheduler with no active split.
    pub fn new() -> Self {
        Self { active_label: None }
    }

    /// Return the split that should receive the next embed step.
    ///
    /// * `newly_selected == true`  — the split was just chosen at a batch
    ///   boundary; the caller should print the announcement banner.
    /// * `newly_selected == false` — continuing an in-progress batch; no
    ///   announcement needed.
    ///
    /// Returns `None` when every split is exhausted or at its limit.
    pub fn next(
        &mut self,
        labels: &[SplitLabel],
        counts: &[u64],
        maxes: &[u64],
        ratios: &[f32],
        exhausted: &[bool],
    ) -> Option<(SplitLabel, bool)> {
        if let Some(label) = self.active_label {
            Some((label, false))
        } else {
            let label = next_split_to_fill(labels, counts, maxes, ratios, exhausted)?;
            self.active_label = Some(label);
            Some((label, true))
        }
    }

    /// Release the lock so the next call to `next()` re-evaluates the split.
    pub fn unlock(&mut self) {
        self.active_label = None;
    }

    /// Returns `true` when a split is currently locked (i.e. `next()` would
    /// return the same split without re-evaluating).
    pub fn is_locked(&self) -> bool {
        self.active_label.is_some()
    }
}

impl Default for SplitScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the number of steps remaining until the next scheduled flush.
///
/// The flush cadence is every `steps_per_batch` steps.  Returns a value in
/// `[1, steps_per_batch]`.
pub fn steps_until_next_flush(step_num: u64, steps_per_batch: u64) -> u64 {
    steps_per_batch - (step_num % steps_per_batch)
}

/// Computes the instantaneous throughput for the current stint.
pub fn compute_samples_per_sec(new_samples: u64, elapsed_secs: f64) -> f64 {
    if elapsed_secs > 0.0 {
        new_samples as f64 / elapsed_secs
    } else {
        0.0
    }
}

/// Substitutes the stale per-iteration snapshot count for `label_pos` with
/// the freshly computed `current_in_flight` and returns the corrected global total.
pub fn compute_global_in_flight(counts: &[u64], label_pos: usize, current_in_flight: u64) -> u64 {
    counts.iter().sum::<u64>() - counts[label_pos] + current_in_flight
}

/// Computes how far a split is behind (or ahead of) its proportional fair
/// share of the global total, and returns a display string.
pub fn compute_deficit_str(
    in_flight: u64,
    global_in_flight: u64,
    ratio: f32,
    ratio_sum: f32,
) -> String {
    let fair_share = if ratio_sum > 0.0 {
        (global_in_flight as f64 * ratio as f64 / ratio_sum as f64).ceil() as u64
    } else {
        0
    };
    let deficit = fair_share.saturating_sub(in_flight);
    if deficit > 0 {
        format!("+{} behind", deficit)
    } else {
        "on target".to_string()
    }
}

/// Returns `true` when the in-flight sample count has reached the per-split cap.
/// `max == 0` means unlimited — never at limit.
pub fn at_sample_limit(max: u64, total_written: u64, pending_len: u64) -> bool {
    max > 0 && total_written + pending_len >= max
}

/// Returns `true` when the current step should trigger a flush-to-disk.
pub fn should_flush_now(
    step_num: u64,
    steps_per_batch: u64,
    ctrl_c: bool,
    hit_limit: bool,
) -> bool {
    step_num.is_multiple_of(steps_per_batch) || ctrl_c || hit_limit
}

// TODO: This needs to be reworked. This is a major hack.
/// Returns `true` when a sampler error message indicates normal split
/// exhaustion rather than a genuine error.
pub fn is_exhaustion_error(msg: &str) -> bool {
    msg.contains("exhausted")
        || msg.contains("no more")
        || msg.contains("empty")
        || msg.contains("no eligible")
}

/// Scale a max sample count proportionally based on split ratios.
pub fn scale_max(max_train_samples: u64, ratio: f32, train_ratio: f32) -> u64 {
    if max_train_samples == 0 {
        return 0;
    }
    ((max_train_samples as f64 * ratio as f64 / train_ratio as f64).ceil() as u64).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_split_to_fill_skips_exhausted() {
        let labels = [SplitLabel::Train, SplitLabel::Validation];
        let counts = [0, 0];
        let maxes = [100, 100];
        let ratios = [0.8, 0.2];
        let exhausted = [true, false];
        assert_eq!(
            next_split_to_fill(&labels, &counts, &maxes, &ratios, &exhausted),
            Some(SplitLabel::Validation)
        );
    }

    #[test]
    fn next_split_to_fill_skips_at_limit() {
        let labels = [SplitLabel::Train, SplitLabel::Validation];
        let counts = [100, 0];
        let maxes = [100, 100];
        let ratios = [0.8, 0.2];
        let exhausted = [false, false];
        assert_eq!(
            next_split_to_fill(&labels, &counts, &maxes, &ratios, &exhausted),
            Some(SplitLabel::Validation)
        );
    }

    #[test]
    fn next_split_to_fill_returns_none_when_all_done() {
        let labels = [SplitLabel::Train, SplitLabel::Validation];
        let counts = [100, 100];
        let maxes = [100, 100];
        let ratios = [0.8, 0.2];
        let exhausted = [false, false];
        assert_eq!(
            next_split_to_fill(&labels, &counts, &maxes, &ratios, &exhausted),
            None
        );
    }

    #[test]
    fn next_split_to_fill_picks_most_behind() {
        let labels = [SplitLabel::Train, SplitLabel::Validation];
        let counts = [50, 10];
        let maxes = [100, 100];
        let ratios = [0.8, 0.2];
        let exhausted = [false, false];
        assert_eq!(
            next_split_to_fill(&labels, &counts, &maxes, &ratios, &exhausted),
            Some(SplitLabel::Validation)
        );
    }

    #[test]
    fn scheduler_locks_and_unlocks() {
        let labels = [SplitLabel::Train, SplitLabel::Validation];
        let counts = [0, 0];
        let maxes = [100, 100];
        let ratios = [0.8, 0.2];
        let exhausted = [false, false];

        let mut sched = SplitScheduler::new();
        let (label, newly) = sched
            .next(&labels, &counts, &maxes, &ratios, &exhausted)
            .unwrap();
        assert!(newly);
        assert_eq!(label, SplitLabel::Train);

        let (label, newly) = sched
            .next(&labels, &counts, &maxes, &ratios, &exhausted)
            .unwrap();
        assert!(!newly);
        assert_eq!(label, SplitLabel::Train);

        sched.unlock();
        let (label, _) = sched
            .next(&labels, &counts, &maxes, &ratios, &exhausted)
            .unwrap();
        assert_eq!(label, SplitLabel::Train);
    }

    #[test]
    fn steps_until_next_flush_basic() {
        assert_eq!(steps_until_next_flush(0, 100), 100);
        assert_eq!(steps_until_next_flush(1, 100), 99);
        assert_eq!(steps_until_next_flush(99, 100), 1);
    }

    #[test]
    fn should_flush_now_at_boundary() {
        assert!(should_flush_now(100, 100, false, false));
        assert!(!should_flush_now(50, 100, false, false));
        assert!(should_flush_now(50, 100, true, false));
        assert!(should_flush_now(50, 100, false, true));
    }

    #[test]
    fn at_sample_limit_zero_means_unlimited() {
        assert!(!at_sample_limit(0, 0, 100));
    }

    #[test]
    fn at_sample_limit_at_cap() {
        assert!(at_sample_limit(100, 90, 10));
    }

    #[test]
    fn is_exhaustion_error_detects_messages() {
        assert!(is_exhaustion_error("sampler exhausted"));
        assert!(is_exhaustion_error("no more data"));
        assert!(is_exhaustion_error("source empty"));
        assert!(is_exhaustion_error("no eligible records"));
        assert!(!is_exhaustion_error("connection refused"));
    }

    #[test]
    fn compute_samples_per_sec_basic() {
        assert_eq!(compute_samples_per_sec(100, 10.0), 10.0);
        assert_eq!(compute_samples_per_sec(100, 0.0), 0.0);
    }

    #[test]
    fn compute_deficit_str_behind() {
        let s = compute_deficit_str(10, 100, 0.8, 1.0);
        assert!(s.contains("behind"));
    }

    #[test]
    fn compute_deficit_str_on_target() {
        let s = compute_deficit_str(75, 100, 0.75, 1.0);
        assert_eq!(s, "on target", "got: {s}");
    }

    #[test]
    fn compute_global_in_flight_substitutes_stale_count() {
        let counts = vec![800u64, 100u64];
        assert_eq!(compute_global_in_flight(&counts, 0, 864), 964);
    }

    #[test]
    fn compute_global_in_flight_val_position() {
        let counts = vec![800u64, 100u64];
        assert_eq!(compute_global_in_flight(&counts, 1, 164), 964);
    }

    #[test]
    fn compute_global_in_flight_single_split() {
        let counts = vec![500u64];
        assert_eq!(compute_global_in_flight(&counts, 0, 564), 564);
    }

    #[test]
    fn scale_max_unlimited_when_zero() {
        assert_eq!(scale_max(0, 0.8, 0.8), 0);
    }

    #[test]
    fn scale_max_proportional() {
        assert_eq!(scale_max(1000, 0.2, 0.8), 250);
    }

    #[test]
    fn scale_max_rounds_up() {
        assert_eq!(scale_max(100, 0.1, 0.8), 13);
    }

    #[test]
    fn scale_max_clamps_to_one() {
        assert_eq!(scale_max(1, 0.1, 0.8), 1);
    }
}
