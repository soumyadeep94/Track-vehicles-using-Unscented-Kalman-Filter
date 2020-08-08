#include "ukf.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 5.0;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.8;

    /**
     * DO NOT MODIFY measurement noise values below.
     * These are provided by the sensor manufacturer.
     */

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;


    // Initially set to false, set to true in first call of ProcessMeasurement
    is_initialized_ = false;

    // State dimension
    n_x_ = 5;

    // Augmented state dimension
    n_aug_ = 7;

    // Sigma point spreading parameter
    lambda_ = 3 - n_x_;

    // initial state vector
    x_ = VectorXd::Zero(n_x_);

    // initial covariance matrix
    P_ = MatrixXd::Zero(n_x_, n_x_);

    // predicted sigma points matrix
    Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

    // Weights of sigma points
    weights_ = VectorXd::Zero(2 * n_aug_ + 1);

    //measurement noise covariance matrix
    R_radar_ = MatrixXd::Zero(3, 3);
    R_radar_ << std_radr_ * std_radr_, 0, 0,
                0, std_radphi_ * std_radphi_ , 0,
                0, 0, std_radrd_*std_radrd_;

    R_laser_ = MatrixXd::Zero(2, 2);
    R_laser_ <<  std_laspx_ * std_laspx_, 0 ,
                  0 , std_laspy_ * std_laspy_ ;

    NIS_Radar_ = NIS_Laser_ = 0;

    prev_time_ = 0;

}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) {
        return;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_) {
        return;
    }
    if (!is_initialized_) {
        /**
         * Initialize the state with the first measurement.
         */
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            // Convert radar from polar to cartesian coordinates and initialize state
            float rho = static_cast<float>(meas_package.raw_measurements_(0));
            float phi = static_cast<float>(meas_package.raw_measurements_(1));
            x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            // Initialize state
            // Set the state with the initial location and zero velocity
            x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
        }

        prev_time_ = meas_package.timestamp_;
        // Done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    auto dt = static_cast<double>((meas_package.timestamp_ - prev_time_) * 1e-6);

    prev_time_ = meas_package.timestamp_;

    while (dt > 0.1) {
        double delta_t = 0.05;
        Prediction(delta_t);
        dt -= delta_t;
    }
    Prediction(dt);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        //set measurement dimension for radar; rho, phi, phi_dot
        n_z_ = 3;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        //set measurement dimension for laser; px, py
        n_z_ = 2;
    }

    //create matrix for sigma points in measurement space
    Zsig_ = MatrixXd::Zero(n_z_, 2 * n_aug_ + 1);

    //mean predicted measurement
    Z_pred_ = VectorXd::Zero(n_z_);

    //measurement covariance matrix S
    S_ = MatrixXd::Zero(n_z_, n_z_);


    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        PredictRadarMeasurement();
        // update NIS
        NIS_Radar_ = (meas_package.raw_measurements_-Z_pred_).transpose()*S_.inverse()*(meas_package.raw_measurements_-Z_pred_);
        std::cout << "NIS Radar: " << NIS_Radar_ << std::endl;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        PredictLaserMeasurement();
        // update NIS
        NIS_Laser_ = (meas_package.raw_measurements_-Z_pred_).transpose()*S_.inverse()*(meas_package.raw_measurements_-Z_pred_);
        std::cout << "NIS LiDAR: " << NIS_Laser_ << std::endl;
    }

    UpdateState(meas_package.raw_measurements_);

}

void UKF::Prediction(double dt) {

    MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);
    CreateAugmentedSigmaPoints(Xsig_aug);
    PredictSigmaPoints(Xsig_aug, dt);
    PredictMeanCov();

}


void UKF::CreateAugmentedSigmaPoints(MatrixXd& Xsig_aug) {

    VectorXd x_aug = VectorXd::Zero(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

    //create augmented mean state
    x_aug.head(n_x_) = x_;

    //create augmented covariance matrix
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a_ * std_a_;
    P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

    // Cholesky decomposition
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i< n_aug_; i++) {
        Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
    }
}


void UKF::PredictSigmaPoints(MatrixXd& Xsig_aug, double dt) {
    //predict sigma points
    for (int i = 0; i< 2*n_aug_+1; i++) {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*dt) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*dt) );
        }
        else {
            px_p = p_x + v*dt*cos(yaw);
            py_p = p_y + v*dt*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*dt;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5*nu_a*dt*dt * cos(yaw);
        py_p = py_p + 0.5*nu_a*dt*dt * sin(yaw);
        v_p = v_p + nu_a*dt;

        yaw_p = yaw_p + 0.5*nu_yawdd*dt*dt;
        yawd_p = yawd_p + nu_yawdd*dt;

        //write predicted sigma point into right column
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }
}


void UKF::PredictMeanCov() {
    // set weights and these weights will be shared during update
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
        double weight = 0.5 / (lambda_ + n_aug_);
        weights_(i) = weight;
    }

    //predicted state mean
    x_.fill(0.0);
    //iterate over sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    //iterate over sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        //angle normalization
        while (x_diff(3)> M_PI) {
            x_diff(3) -= 2. * M_PI;
        }
        while (x_diff(3)<-M_PI) {
            x_diff(3) += 2. * M_PI;
        }

        P_ += weights_(i) * x_diff * x_diff.transpose() ;
    }
}


void UKF::PredictRadarMeasurement() {
    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

        // extract values for better readability
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v   = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // measurement model
        Zsig_(0,i) = sqrt(p_x*p_x  +  p_y*p_y);                   //r
        Zsig_(1,i) = atan2(p_y, p_x);                            //phi

        if (Zsig_(0, i) < 0.001) {
            Zsig_(2, i) = (p_x * v1 + p_y * v2) / 0.001;        //r_dot
        }
        else {
            Zsig_(2, i) = (p_x * v1 + p_y * v2) / Zsig_(0, i);  //r_dot;
        }
    }

    Z_pred_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        Z_pred_ = Z_pred_ + weights_(i) * Zsig_.col(i);
    }

    S_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        //residual
        VectorXd z_diff = Zsig_.col(i) - Z_pred_;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
        while (z_diff(1) <- M_PI) z_diff(1) += 2.*M_PI;

        S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
    }

    S_ = S_ + R_radar_;
}

void UKF::PredictLaserMeasurement() {
    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        // measurement model
        Zsig_(0,i) = Xsig_pred_(0,i);           //px
        Zsig_(1,i) = Xsig_pred_(1,i);           //py
    }

    Z_pred_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        Z_pred_ = Z_pred_ + weights_(i) * Zsig_.col(i);
    }


    S_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        //residual
        VectorXd z_diff = Zsig_.col(i) - Z_pred_;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

        S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
    }

    S_ = S_ + R_laser_;
}


void UKF::UpdateState(const VectorXd& z) {
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_);

    //calculate cross correlation matrix
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

        //residual
        VectorXd z_diff = Zsig_.col(i) - Z_pred_;
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S_.inverse();

    //residual
    VectorXd z_diff = z - Z_pred_;

    //angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;

    while (z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S_*K.transpose();
}