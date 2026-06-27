use std::cmp::Ordering;
use std::slice;
use std::thread;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SearchCandidate {
    pub start_x_m: f64,
    pub start_y_m: f64,
    pub speed_mps: f64,
    pub azimuth_deg: f64,
    pub error_rmse_m: f64,
    pub correlation: f64,
    pub speed_index: usize,
    pub azimuth_index: usize,
}

impl Default for SearchCandidate {
    fn default() -> Self {
        Self {
            start_x_m: 0.0,
            start_y_m: 0.0,
            speed_mps: 0.0,
            azimuth_deg: 0.0,
            error_rmse_m: f64::INFINITY,
            correlation: f64::NAN,
            speed_index: 0,
            azimuth_index: 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn terrain_nav_search(
    dem_ptr: *const f64,
    rows: usize,
    cols: usize,
    origin_x_m: f64,
    origin_y_m: f64,
    pixel_width_m: f64,
    pixel_height_m: f64,
    measured_ptr: *const f64,
    timestamps_ptr: *const f64,
    point_count: usize,
    starts_ptr: *const f64,
    start_count: usize,
    speeds_ptr: *const f64,
    speed_count: usize,
    azimuths_ptr: *const f64,
    azimuth_count: usize,
    use_weighted_scoring: i32,
    top_k: usize,
    out_candidates_ptr: *mut SearchCandidate,
    out_count_ptr: *mut usize,
) -> i32 {
    if dem_ptr.is_null()
        || measured_ptr.is_null()
        || timestamps_ptr.is_null()
        || starts_ptr.is_null()
        || speeds_ptr.is_null()
        || azimuths_ptr.is_null()
        || out_candidates_ptr.is_null()
        || out_count_ptr.is_null()
        || rows < 2
        || cols < 2
        || point_count == 0
        || start_count == 0
        || speed_count == 0
        || azimuth_count == 0
        || top_k == 0
        || pixel_width_m == 0.0
        || pixel_height_m == 0.0
    {
        return -1;
    }

    let dem = unsafe { slice::from_raw_parts(dem_ptr, rows * cols) };
    let measured = unsafe { slice::from_raw_parts(measured_ptr, point_count) };
    let timestamps = unsafe { slice::from_raw_parts(timestamps_ptr, point_count) };
    let starts = unsafe { slice::from_raw_parts(starts_ptr, start_count * 2) };
    let speeds = unsafe { slice::from_raw_parts(speeds_ptr, speed_count) };
    let azimuths = unsafe { slice::from_raw_parts(azimuths_ptr, azimuth_count) };
    let weights = if use_weighted_scoring != 0 {
        make_profile_weights(measured)
    } else {
        vec![1.0; measured.len()]
    };

    let workers = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(start_count)
        .max(1);
    let chunk_size = (start_count + workers - 1) / workers;
    let mut merged: Vec<SearchCandidate> = Vec::with_capacity(top_k);

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker_index in 0..workers {
            let begin = worker_index * chunk_size;
            let end = ((worker_index + 1) * chunk_size).min(start_count);
            if begin >= end {
                continue;
            }

            let weights_ref = weights.as_slice();
            handles.push(scope.spawn(move || {
                let mut local: Vec<SearchCandidate> = Vec::with_capacity(top_k);
                for start_index in begin..end {
                    let start_x_m = starts[start_index * 2];
                    let start_y_m = starts[start_index * 2 + 1];
                    for (speed_index, speed_mps) in speeds.iter().copied().enumerate() {
                        for (azimuth_index, azimuth_deg) in azimuths.iter().copied().enumerate() {
                            if let Some((rmse, corr)) = evaluate_trajectory(
                                dem,
                                rows,
                                cols,
                                origin_x_m,
                                origin_y_m,
                                pixel_width_m,
                                pixel_height_m,
                                measured,
                                weights_ref,
                                timestamps,
                                start_x_m,
                                start_y_m,
                                speed_mps,
                                azimuth_deg,
                                local.last().map(candidate_score).unwrap_or(f64::INFINITY),
                            ) {
                                push_candidate(
                                    &mut local,
                                    SearchCandidate {
                                        start_x_m,
                                        start_y_m,
                                        speed_mps,
                                        azimuth_deg,
                                        error_rmse_m: rmse,
                                        correlation: corr,
                                        speed_index,
                                        azimuth_index,
                                    },
                                    top_k,
                                );
                            }
                        }
                    }
                }
                local
            }));
        }

        for handle in handles {
            if let Ok(local) = handle.join() {
                for candidate in local {
                    push_candidate(&mut merged, candidate, top_k);
                }
            }
        }
    });

    let out_count = merged.len().min(top_k);
    let out_candidates = unsafe { slice::from_raw_parts_mut(out_candidates_ptr, top_k) };
    for item in out_candidates.iter_mut() {
        *item = SearchCandidate::default();
    }
    for index in 0..out_count {
        out_candidates[index] = merged[index];
    }
    unsafe {
        *out_count_ptr = out_count;
    }

    if out_count == 0 {
        return 1;
    }
    0
}

#[no_mangle]
pub extern "C" fn terrain_nav_error_grid(
    dem_ptr: *const f64,
    rows: usize,
    cols: usize,
    origin_x_m: f64,
    origin_y_m: f64,
    pixel_width_m: f64,
    pixel_height_m: f64,
    measured_ptr: *const f64,
    timestamps_ptr: *const f64,
    point_count: usize,
    start_x_m: f64,
    start_y_m: f64,
    speeds_ptr: *const f64,
    speed_count: usize,
    azimuths_ptr: *const f64,
    azimuth_count: usize,
    use_weighted_scoring: i32,
    out_errors_ptr: *mut f64,
    out_corrs_ptr: *mut f64,
) -> i32 {
    if dem_ptr.is_null()
        || measured_ptr.is_null()
        || timestamps_ptr.is_null()
        || speeds_ptr.is_null()
        || azimuths_ptr.is_null()
        || out_errors_ptr.is_null()
        || out_corrs_ptr.is_null()
        || rows < 2
        || cols < 2
        || point_count == 0
        || speed_count == 0
        || azimuth_count == 0
        || pixel_width_m == 0.0
        || pixel_height_m == 0.0
    {
        return -1;
    }

    let dem = unsafe { slice::from_raw_parts(dem_ptr, rows * cols) };
    let measured = unsafe { slice::from_raw_parts(measured_ptr, point_count) };
    let timestamps = unsafe { slice::from_raw_parts(timestamps_ptr, point_count) };
    let speeds = unsafe { slice::from_raw_parts(speeds_ptr, speed_count) };
    let azimuths = unsafe { slice::from_raw_parts(azimuths_ptr, azimuth_count) };
    let errors = unsafe { slice::from_raw_parts_mut(out_errors_ptr, speed_count * azimuth_count) };
    let corrs = unsafe { slice::from_raw_parts_mut(out_corrs_ptr, speed_count * azimuth_count) };
    let weights = if use_weighted_scoring != 0 {
        make_profile_weights(measured)
    } else {
        vec![1.0; measured.len()]
    };

    for (speed_index, speed_mps) in speeds.iter().copied().enumerate() {
        for (azimuth_index, azimuth_deg) in azimuths.iter().copied().enumerate() {
            let flat_index = speed_index * azimuth_count + azimuth_index;
            if let Some((rmse, corr)) = evaluate_trajectory(
                dem,
                rows,
                cols,
                origin_x_m,
                origin_y_m,
                pixel_width_m,
                pixel_height_m,
                measured,
                &weights,
                timestamps,
                start_x_m,
                start_y_m,
                speed_mps,
                azimuth_deg,
                f64::INFINITY,
            ) {
                errors[flat_index] = rmse;
                corrs[flat_index] = corr;
            } else {
                errors[flat_index] = f64::INFINITY;
                corrs[flat_index] = f64::NAN;
            }
        }
    }

    0
}

fn evaluate_trajectory(
    dem: &[f64],
    rows: usize,
    cols: usize,
    origin_x_m: f64,
    origin_y_m: f64,
    pixel_width_m: f64,
    pixel_height_m: f64,
    measured: &[f64],
    weights: &[f64],
    timestamps: &[f64],
    start_x_m: f64,
    start_y_m: f64,
    speed_mps: f64,
    azimuth_deg: f64,
    rmse_abort_limit: f64,
) -> Option<(f64, f64)> {
    let azimuth_rad = azimuth_deg.to_radians();
    let sin_az = azimuth_rad.sin();
    let cos_az = azimuth_rad.cos();
    let t0 = timestamps[0];
    let total_weight: f64 = weights.iter().copied().filter(|value| value.is_finite()).sum();
    if total_weight <= 1e-9 || weights.len() != measured.len() {
        return None;
    }

    let mut weighted_squared_error_sum = 0.0;
    let mut sum_measured = 0.0;
    let mut sum_predicted = 0.0;
    let mut sum_measured2 = 0.0;
    let mut sum_predicted2 = 0.0;
    let mut sum_cross = 0.0;

    for index in 0..measured.len() {
        let measured_height = measured[index];
        let weight = weights[index];
        if !measured_height.is_finite() || !weight.is_finite() || weight <= 0.0 {
            return None;
        }

        let distance_m = speed_mps * (timestamps[index] - t0);
        let x_m = start_x_m + distance_m * sin_az;
        let y_m = start_y_m + distance_m * cos_az;
        let predicted_height = sample_bilinear(
            dem,
            rows,
            cols,
            origin_x_m,
            origin_y_m,
            pixel_width_m,
            pixel_height_m,
            x_m,
            y_m,
        )?;

        let diff = predicted_height - measured_height;
        weighted_squared_error_sum += weight * diff * diff;
        if rmse_abort_limit.is_finite() && index % 32 == 31 {
            let best_possible_rmse = (weighted_squared_error_sum / total_weight).sqrt();
            if best_possible_rmse > rmse_abort_limit {
                return None;
            }
        }

        sum_measured += weight * measured_height;
        sum_predicted += weight * predicted_height;
        sum_measured2 += weight * measured_height * measured_height;
        sum_predicted2 += weight * predicted_height * predicted_height;
        sum_cross += weight * measured_height * predicted_height;
    }

    let rmse = (weighted_squared_error_sum / total_weight).sqrt();
    let covariance = total_weight * sum_cross - sum_measured * sum_predicted;
    let measured_var = total_weight * sum_measured2 - sum_measured * sum_measured;
    let predicted_var = total_weight * sum_predicted2 - sum_predicted * sum_predicted;
    let denominator = (measured_var * predicted_var).sqrt();
    let correlation = if denominator > 1e-9 {
        (covariance / denominator).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    Some((rmse, correlation))
}

fn make_profile_weights(measured: &[f64]) -> Vec<f64> {
    if measured.len() < 3 {
        return vec![1.0; measured.len()];
    }

    let mut features = vec![0.0; measured.len()];
    for index in 0..measured.len() {
        let current = measured[index];
        if !current.is_finite() {
            continue;
        }

        let prev = profile_value(measured, index.saturating_sub(1), current);
        let next = profile_value(measured, (index + 1).min(measured.len() - 1), current);
        let gradient = (next - prev).abs() * 0.5;
        let curvature = (next - 2.0 * current + prev).abs();
        features[index] = gradient + 0.5 * curvature;
    }

    let feature_sum: f64 = features.iter().copied().filter(|value| value.is_finite()).sum();
    let feature_mean = feature_sum / measured.len() as f64;
    if !feature_mean.is_finite() || feature_mean <= 1e-9 {
        return vec![1.0; measured.len()];
    }

    features
        .iter()
        .map(|feature| 1.0 + (feature / feature_mean).clamp(0.0, 4.0))
        .collect()
}

fn profile_value(values: &[f64], index: usize, fallback: f64) -> f64 {
    let value = values[index];
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn sample_bilinear(
    dem: &[f64],
    rows: usize,
    cols: usize,
    origin_x_m: f64,
    origin_y_m: f64,
    pixel_width_m: f64,
    pixel_height_m: f64,
    x_m: f64,
    y_m: f64,
) -> Option<f64> {
    let col = (x_m - origin_x_m) / pixel_width_m - 0.5;
    let row = (y_m - origin_y_m) / pixel_height_m - 0.5;

    if !row.is_finite()
        || !col.is_finite()
        || row < 0.0
        || col < 0.0
        || row >= (rows - 1) as f64
        || col >= (cols - 1) as f64
    {
        return None;
    }

    let row0 = row.floor() as usize;
    let col0 = col.floor() as usize;
    let dr = row - row0 as f64;
    let dc = col - col0 as f64;

    let idx00 = row0 * cols + col0;
    let idx10 = (row0 + 1) * cols + col0;
    let h00 = dem[idx00];
    let h01 = dem[idx00 + 1];
    let h10 = dem[idx10];
    let h11 = dem[idx10 + 1];

    if !h00.is_finite() || !h01.is_finite() || !h10.is_finite() || !h11.is_finite() {
        return None;
    }

    let top = h00 * (1.0 - dc) + h01 * dc;
    let bottom = h10 * (1.0 - dc) + h11 * dc;
    Some(top * (1.0 - dr) + bottom * dr)
}

fn push_candidate(top: &mut Vec<SearchCandidate>, candidate: SearchCandidate, top_k: usize) {
    if !candidate.error_rmse_m.is_finite() {
        return;
    }

    if top.len() < top_k {
        top.push(candidate);
        top.sort_by(compare_candidates);
        return;
    }

    if let Some(worst) = top.last() {
        if compare_candidates(&candidate, worst) == Ordering::Less {
            top.pop();
            top.push(candidate);
            top.sort_by(compare_candidates);
        }
    }
}

fn compare_candidates(left: &SearchCandidate, right: &SearchCandidate) -> Ordering {
    candidate_score(left)
        .total_cmp(&candidate_score(right))
        .then_with(|| left.error_rmse_m.total_cmp(&right.error_rmse_m))
        .then_with(|| right.correlation.total_cmp(&left.correlation))
}

fn candidate_score(candidate: &SearchCandidate) -> f64 {
    if !candidate.error_rmse_m.is_finite() {
        return f64::INFINITY;
    }

    let corr = if candidate.correlation.is_finite() {
        candidate.correlation.clamp(-1.0, 1.0)
    } else {
        -1.0
    };
    let correlation_penalty = (1.0 - corr).max(0.0);
    candidate.error_rmse_m * (1.0 + 0.35 * correlation_penalty)
}
