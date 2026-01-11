#ifndef POSTURE_CALIBRATION_H
#define POSTURE_CALIBRATION_H

#include "qmi8658.h"
#include <stdbool.h>
#include <stdint.h>

// ============================================================
// CONFIGURATION
// ============================================================
#define CAL_SAMPLE_COUNT        2000     // At 1000Hz = 2 seconds
#define CAL_ACCEL_MOTION_THRESH 50.0f    // Max deviation in mg during calibration
#define CAL_GYRO_MOTION_THRESH  5.0f     // Max rotation rate °/s during calibration
#define CAL_GRAVITY_MG          1000.0f  // Expected gravity magnitude in mg

// ============================================================
// TYPES
// ============================================================
typedef enum {
    CAL_STATUS_IDLE,
    CAL_STATUS_COLLECTING,
    CAL_STATUS_SUCCESS,
    CAL_STATUS_FAILED_MOTION,
    CAL_STATUS_FAILED_ORIENTATION
} cal_status_t;

typedef struct {
    // Gyroscope bias (subtract from raw readings)
    float gyro_bias_x;      // °/s
    float gyro_bias_y;      // °/s
    float gyro_bias_z;      // °/s
    
    // Accelerometer reference in neutral posture
    float accel_ref_x;      // mg
    float accel_ref_y;      // mg
    float accel_ref_z;      // mg
    
    // Neutral posture angles (calculated from accel reference)
    float neutral_pitch;    // degrees
    float neutral_roll;     // degrees
    
    // Validity flag
    bool is_valid;
} posture_cal_data_t;

typedef struct {
    // Accumulators
    double sum_ax, sum_ay, sum_az;
    double sum_gx, sum_gy, sum_gz;
    
    // Motion detection (min/max tracking)
    float min_ax, max_ax;
    float min_ay, max_ay;
    float min_az, max_az;
    float max_abs_gx, max_abs_gy, max_abs_gz;
    
    // Progress
    uint16_t sample_count;
    cal_status_t status;
} posture_cal_state_t;

// ============================================================
// API
// ============================================================

/**
 * @brief Start calibration process
 * @param state Calibration state to initialize
 */
void posture_cal_start(posture_cal_state_t *state);

/**
 * @brief Feed a sample during calibration
 * @param state Calibration state
 * @param data  Pointer to QMI8658 sensor data (from qmi_data)
 * @note Call this at sensor ODR while status == CAL_STATUS_COLLECTING
 */
void posture_cal_add_sample(posture_cal_state_t *state, const qmi8658_data_t *data);

/**
 * @brief Finalize calibration and compute results
 * @param state Calibration state
 * @param cal   Output calibration data (populated on success)
 * @return true if calibration succeeded
 */
bool posture_cal_finish(posture_cal_state_t *state, posture_cal_data_t *cal);

/**
 * @brief Check if still collecting samples
 * @param state Calibration state
 * @return true if calibration is in progress
 */
bool posture_cal_is_collecting(const posture_cal_state_t *state);

/**
 * @brief Get calibration progress
 * @param state Calibration state
 * @return Progress percentage (0-100)
 */
uint8_t posture_cal_progress(const posture_cal_state_t *state);

/**
 * @brief Get calibration status
 * @param state Calibration state
 * @return Current status enum
 */
cal_status_t posture_cal_get_status(const posture_cal_state_t *state);

/**
 * @brief Apply calibration to get corrected gyro readings
 * @param cal      Calibration data
 * @param raw_gx   Raw gyro X (°/s)
 * @param raw_gy   Raw gyro Y (°/s)
 * @param raw_gz   Raw gyro Z (°/s)
 * @param cor_gx   Output corrected gyro X
 * @param cor_gy   Output corrected gyro Y
 * @param cor_gz   Output corrected gyro Z
 */
void posture_cal_apply_gyro(
    const posture_cal_data_t *cal,
    float raw_gx, float raw_gy, float raw_gz,
    float *cor_gx, float *cor_gy, float *cor_gz
);

/**
 * @brief Calculate pitch/roll relative to calibrated neutral
 * @param cal       Calibration data
 * @param ax        Accelerometer X (mg)
 * @param ay        Accelerometer Y (mg)
 * @param az        Accelerometer Z (mg)
 * @param pitch_deg Output pitch angle (degrees, + = forward lean)
 * @param roll_deg  Output roll angle (degrees, + = right tilt)
 */
void posture_cal_get_angles(
    const posture_cal_data_t *cal,
    float ax, float ay, float az,
    float *pitch_deg, float *roll_deg
);

#endif // POSTURE_CALIBRATION_H