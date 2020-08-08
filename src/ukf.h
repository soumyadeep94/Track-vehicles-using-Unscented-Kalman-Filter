#pragma once

#include <iostream>
#include "Eigen/Dense"
#include "measurement_package.h"

class UKF {
public:
    /**
     * Constructor
     */
    UKF();

    /**
     * Destructor
     */
    virtual ~UKF() = default;

    /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
    void ProcessMeasurement(MeasurementPackage meas_package);

    /**
     * Prediction Predicts sigma points, the state, and the state covariance
     * matrix
     * @param dt Time between k and k+1 in s
     */
    void Prediction(double delta_t);

    void CreateAugmentedSigmaPoints(Eigen::MatrixXd& Xsig_out);
    void PredictSigmaPoints(Eigen::MatrixXd& Xsig_aug, double dt);
    void PredictMeanCov();
    void PredictRadarMeasurement();
    void PredictLaserMeasurement();
    void UpdateState(const Eigen::VectorXd& z);


    // initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;

    // if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    // if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    Eigen::VectorXd x_;

    // state covariance matrix
    Eigen::MatrixXd P_;

    // predicted sigma points matrix
    Eigen::MatrixXd Xsig_pred_;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    // Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    // Laser measurement noise standard deviation position1 in m
    double std_laspx_;

    // Laser measurement noise standard deviation position2 in m
    double std_laspy_;

    // Radar measurement noise standard deviation radius in m
    double std_radr_;

    // Radar measurement noise standard deviation angle in rad
    double std_radphi_;

    // Radar measurement noise standard deviation radius change in m/s
    double std_radrd_ ;

    // Weights of sigma points
    Eigen::VectorXd weights_;

    // State dimension
    int n_x_;

    // Augmented state dimension
    int n_aug_;

    // Sigma point spreading parameter
    double lambda_;

    // current NIS for radar
    double NIS_Radar_;

    // current NIS for laser
    double NIS_Laser_;

private:
    // previous timestamp
    long prev_time_;

    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z_;

    //create matrix for sigma points in measurement space
    Eigen::MatrixXd Zsig_;

    //mean predicted measurement
    Eigen::VectorXd Z_pred_;

    //measurement covariance matrix S
    Eigen::MatrixXd S_;

    //measurement noise covariance matrix
    Eigen::MatrixXd R_laser_, R_radar_;

};