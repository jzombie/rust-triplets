//! Deterministic grouping helpers shared by source implementations.
//!
//! This module provides a pure, deterministic re-ordering that keeps related
//! items closer together inside bounded windows while preserving global
//! reproducibility (same inputs + seed => same output).

use std::collections::{HashMap, VecDeque};

use crate::hash::stable_hash_str;
use crate::types::{GroupKey, ItemOrderKey};

/// Compute the per-group upper bound used when selecting a *limited* refresh slice.
///
/// This solves the "dominant directory" problem: when one group has far more
/// records than others (for example `factual/xbrl_definitions`), a refresh with
/// a small `limit` can otherwise be filled almost entirely by that group.
///
/// The returned cap is `ceil(limit / group_count)`, which guarantees a single
/// group cannot contribute more than its fair share of the current refresh.
///
/// Scope note: this cap applies to per-refresh selection (cursor + limit), not
/// to construction of the full deterministic permutation.
///
/// Edge cases:
/// - `limit == 0` => `0`
/// - `group_count == 0` => treated as `1` to avoid division by zero
pub fn per_group_refresh_cap(limit: usize, group_count: usize) -> usize {
    if limit == 0 {
        return 0;
    }
    let group_count = group_count.max(1);
    limit.div_ceil(group_count).max(1)
}

/// Deterministically re-orders `items` within fixed-size windows.
///
/// Steps per window:
/// 1) Split items into windows (size = total / window_divisor, at least 1).
/// 2) Group each window by `group_key`.
/// 3) Deterministically interleave groups to encourage locality.
/// 4) Deterministically shuffle the interleaved pool for mixing.
///
/// `item_key` provides a stable per-item key used for deterministic ordering
/// inside groups and for the final per-window shuffle.
pub fn deterministic_grouped_order<T, FItemKey, FGroupKey>(
    items: &[T],
    seed: u64,
    window_divisor: usize,
    item_key: FItemKey,
    group_key: FGroupKey,
) -> Vec<T>
where
    T: Clone,
    FItemKey: Fn(&T) -> ItemOrderKey,
    FGroupKey: Fn(&T) -> GroupKey,
{
    if items.is_empty() {
        return Vec::new();
    }

    let total = items.len();
    let window_size = (total / window_divisor).max(1);
    if window_size == 1 {
        return items.to_vec();
    }

    // Bucket every item by group so we can enforce per-window caps while still
    // emitting a full permutation of the input.
    let mut grouped_items: HashMap<GroupKey, Vec<T>> = HashMap::new();
    for item in items {
        let key = group_key(item);
        grouped_items.entry(key).or_default().push(item.clone());
    }

    let mut group_keys: Vec<GroupKey> = grouped_items.keys().cloned().collect();
    group_keys.sort_by_key(|key| (stable_hash_str(seed, key), key.clone()));

    let mut group_queues: HashMap<GroupKey, VecDeque<T>> = HashMap::new();
    for key in &group_keys {
        if let Some(mut entries) = grouped_items.remove(key) {
            let group_seed = stable_hash_str(seed, key);
            entries.sort_by_key(|item| {
                let k = item_key(item);
                (stable_hash_str(group_seed, &k), k)
            });
            group_queues.insert(key.clone(), VecDeque::from(entries));
        }
    }

    let mut order = Vec::with_capacity(total);
    let mut remaining = total;

    while remaining > 0 {
        let window_len = remaining.min(window_size);
        let active_keys: Vec<GroupKey> = group_keys
            .iter()
            .filter(|key| {
                group_queues
                    .get(*key)
                    .map(|queue| !queue.is_empty())
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        if active_keys.is_empty() {
            break;
        }

        // Strict cap for this window: each active group can contribute at most
        // ceil(window_len / active_group_count) items in this window.
        let per_group_cap = window_len.div_ceil(active_keys.len()).max(1);
        let mut produced = 0usize;

        for key in active_keys {
            if produced >= window_len {
                break;
            }
            let Some(queue) = group_queues.get_mut(&key) else {
                continue;
            };
            let mut take = per_group_cap.min(window_len - produced);
            while take > 0 {
                if let Some(item) = queue.pop_front() {
                    order.push(item);
                    produced += 1;
                    remaining -= 1;
                    take -= 1;
                    if produced >= window_len {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_grouped_order_is_deterministic() {
        let items = vec![
            "alpha/01".to_string(),
            "beta/02".to_string(),
            "alpha/03".to_string(),
            "beta/04".to_string(),
        ];
        let order_a = deterministic_grouped_order(
            &items,
            42,
            2,
            |item| item.clone(),
            |item| item.split('/').next().unwrap_or("").to_string(),
        );
        let order_b = deterministic_grouped_order(
            &items,
            42,
            2,
            |item| item.clone(),
            |item| item.split('/').next().unwrap_or("").to_string(),
        );
        assert_eq!(order_a, order_b);
    }

    #[test]
    fn deterministic_grouped_order_window_size_one_is_identity() {
        let items = vec![
            "alpha/01".to_string(),
            "beta/02".to_string(),
            "alpha/03".to_string(),
            "beta/04".to_string(),
        ];
        let order = deterministic_grouped_order(
            &items,
            7,
            items.len(),
            |item| item.clone(),
            |item| item.split('/').next().unwrap_or("").to_string(),
        );
        assert_eq!(order, items);
    }

    #[test]
    fn deterministic_grouped_order_returns_permutation() {
        let items = vec![
            "alpha/01".to_string(),
            "beta/02".to_string(),
            "alpha/03".to_string(),
            "beta/04".to_string(),
            "gamma/05".to_string(),
        ];
        let mut order = deterministic_grouped_order(
            &items,
            99,
            2,
            |item| item.clone(),
            |item| item.split('/').next().unwrap_or("").to_string(),
        );
        let mut sorted_items = items.clone();
        sorted_items.sort();
        order.sort();
        assert_eq!(order, sorted_items);
    }

    #[test]
    fn deterministic_grouped_order_caps_per_group_per_window() {
        let mut items = Vec::new();
        for i in 0..40 {
            items.push(format!("large/{i:04}"));
        }
        for i in 0..40 {
            items.push(format!("small/{i:04}"));
        }

        let window_divisor = 2;
        let order = deterministic_grouped_order(
            &items,
            123,
            window_divisor,
            |item| item.clone(),
            |item| item.split('/').next().unwrap_or("").to_string(),
        );

        let window_size = (items.len() / window_divisor).max(1);
        let per_group_cap = window_size.div_ceil(2).max(1);
        let first_window = &order[..window_size.min(order.len())];
        let large_count = first_window
            .iter()
            .filter(|entry| entry.starts_with("large/"))
            .count();
        assert!(large_count <= per_group_cap);
    }

    #[test]
    fn per_group_refresh_cap_uses_ceiling_division() {
        assert_eq!(per_group_refresh_cap(0, 3), 0);
        assert_eq!(per_group_refresh_cap(1, 10), 1);
        assert_eq!(per_group_refresh_cap(9, 3), 3);
        assert_eq!(per_group_refresh_cap(10, 3), 4);
        assert_eq!(per_group_refresh_cap(7, 0), 7);
    }
}
