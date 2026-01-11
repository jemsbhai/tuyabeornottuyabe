#include "posture_calibration.h"
#include <math.h>
#include <string.h>

// ============================================================
// HELPERS
// ============================================================
static inline float rad_to_deg(float rad) {
    return rad * (180.0f / M_PI);
}

static inline float abs_f(float x) {
    return (x < 0.0f) ? -x : x;
}

// ============================================================
// START CALIBRATION
// ============================================================
void posture_cal_start(posture_cal_state_t *state) {
    memset(state, 0, sizeof(posture_cal_state_t));
    
    // Initialize min/max to extreme values
    state->min_ax = 1e9f;   state->max_ax = -1e9f;
    state->min_ay = 1e9f;   state->max_ay = -1e9f;
    state->min_az = 1e9f;   state->max_az = -1e9f;
    
    state->status = CAL_STATUS_COLLECTING;
}

// ============================================================
// ADD SAMPLE
// ============================================================
void posture_cal_add_sample(posture_cal_state_t *state, const qmi8658_data_t *data) {
    if (state->status != CAL_STATUS_COLLECTING) return;
    
    // Accumulate for averaging
    state->sum_ax += data->accelX;
    state->sum_ay += data->accelY;
    state->sum_az += data->accelZ;
    
    state->sum_gx += data->gyroX;
    state->sum_gy += data->gyroY;
    state->sum_gz += data->gyroZ;
    
    // Track accelerometer min/max for motion detection
    if (data->accelX < state->min_ax) state->min_ax = data->accelX;
    if (data->accelX > state->max_ax) state->max_ax = data->accelX;
    if (data->accelY < state->min_ay) state->min_ay = data->accelY;
    if (data->accelY > state->max_ay) state->max_ay = data->accelY;
    if (data->accelZ < state->min_az) state->min_az = data->accelZ;
    if (data->accelZ > state->max_az) state->max_az = data->accelZ;
    
    // Track max gyro magnitude
    float abs_gx = abs_f(data->gyroX);
    float abs_gy = abs_f(data->gyroY);
    float abs_gz = abs_f(data->gyroZ);
    
    if (abs_gx > state->max_abs_gx) state->max_abs_gx = abs_gx;
    if (abs_gy > state->max_abs_gy) state->max_abs_gy = abs_gy;
    if (abs_gz > state->max_abs_gz) state->max_abs_gz = abs_gz;
    
    state->sample_count++;
}

// ============================================================
// FINISH CALIBRATION
// ============================================================
bool posture_cal_finish(posture_cal_state_t *state, posture_cal_data_t *cal) {
    // Check we have enough samples
    if (state->sample_count < CAL_SAMPLE_COUNT) {
        return false;  // Not done yet
    }
    
    // ---------------------------------------------------------
    // Check for motion during calibration
    // ---------------------------------------------------------
    float spread_ax = state->max_ax - state->min_ax;
    float spread_ay = state->max_ay - state->min_ay;
    float spread_az = state->max_az - state->min_az;
    
    bool accel_stable = (spread_ax < CAL_ACCEL_MOTION_THRESH) &&
                        (spread_ay < CAL_ACCEL_MOTION_THRESH) &&
                        (spread_az < CAL_ACCEL_MOTION_THRESH);
    
    bool gyro_stable = (state->max_abs_gx < CAL_GYRO_MOTION_THRESH) &&
                       (state->max_abs_gy < CAL_GYRO_MOTION_THRESH) &&
                       (state->max_abs_gz < CAL_GYRO_MOTION_THRESH);
    
    if (!accel_stable || !gyro_stable) {
        state->status = CAL_STATUS_FAILED_MOTION;
        cal->is_valid = false;
        return false;
    }
    
    // ---------------------------------------------------------
    // Calculate averages
    // ---------------------------------------------------------
    float n = (float)state->sample_count;
    
    cal->accel_ref_x = (float)(state->sum_ax / n);
    cal->accel_ref_y = (float)(state->sum_ay / n);
    cal->accel_ref_z = (float)(state->sum_az / n);
    
    cal->gyro_bias_x = (float)(state->sum_gx / n);
    cal->gyro_bias_y = (float)(state->sum_gy / n);
    cal->gyro_bias_z = (float)(state->sum_gz / n);
    
    // ---------------------------------------------------------
    // Validate accelerometer magnitude (should be ~1000 mg = 1g)
    // ---------------------------------------------------------
    float accel_mag = sqrtf(
        cal->accel_ref_x * cal->accel_ref_x +
        cal->accel_ref_y * cal->accel_ref_y +
        cal->accel_ref_z * cal->accel_ref_z
    );
    
    // Allow 10% tolerance around 1000 mg
    if (accel_mag < (CAL_GRAVITY_MG * 0.9f) || accel_mag > (CAL_GRAVITY_MG * 1.1f)) {
        state->status = CAL_STATUS_FAILED_ORIENTATION;
        cal->is_valid = false;
        return false;
    }
    
    // ---------------------------------------------------------
    // Calculate neutral posture angles
    // Assumes pendant coordinate system where:
    //   +X points forward (out of chest)
    //   +Z points down (with gravity when upright)
    //   +Y points right
    // Adjust atan2 arguments if your board orientation differs
    // ---------------------------------------------------------
    cal->neutral_pitch = rad_to_deg(atan2f(
        cal->accel_ref_x,
        sqrtf(cal->accel_ref_y * cal->accel_ref_y + cal->accel_ref_z * cal->accel_ref_z)
    ));
    
    cal->neutral_roll = rad_to_deg(atan2f(cal->accel_ref_y, cal->accel_ref_z));
    
    // ---------------------------------------------------------
    // Success
    // ---------------------------------------------------------
    cal->is_valid = true;
    state->status = CAL_STATUS_SUCCESS;
    
    return true;
}

// ============================================================
// STATUS HELPERS
// ============================================================
bool posture_cal_is_collecting(const posture_cal_state_t *state) {
    return (state->status == CAL_STATUS_COLLECTING) &&
           (state->sample_count < CAL_SAMPLE_COUNT);
}

uint8_t posture_cal_progress(const posture_cal_state_t *state) {
    if (state->sample_count >= CAL_SAMPLE_COUNT) return 100;
    return (uint8_t)((state->sample_count * 100) / CAL_SAMPLE_COUNT);
}

cal_status_t posture_cal_get_status(const posture_cal_state_t *state) {
    return state->status;
}

// ============================================================
// APPLY CALIBRATION
// ============================================================
void posture_cal_apply_gyro(
    const posture_cal_data_t *cal,
    float raw_gx, float raw_gy, float raw_gz,
    float *cor_gx, float *cor_gy, float *cor_gz
) {
    *cor_gx = raw_gx - cal->gyro_bias_x;
    *cor_gy = raw_gy - cal->gyro_bias_y;
    *cor_gz = raw_gz - cal->gyro_bias_z;
}

void posture_cal_get_angles(
    const posture_cal_data_t *cal,
    float ax, float ay, float az,
    float *pitch_deg, float *roll_deg
) {
    // Current absolute angles
    float current_pitch = rad_to_deg(atan2f(
        ax,
        sqrtf(ay * ay + az * az)
    ));
    
    float current_roll = rad_to_deg(atan2f(ay, az));
    
    // Subtract neutral reference to get relative angles
    *pitch_deg = current_pitch - cal->neutral_pitch;
    *roll_deg = current_roll - cal->neutral_roll;
}