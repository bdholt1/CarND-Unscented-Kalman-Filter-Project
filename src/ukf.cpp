#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  // start with an uninitialised system
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // state dimension
  n_x_ = 5;

  // augmented state dimension
  n_aug_ = 7;

  // number of sigma points
  n_sigma_ = 2 * n_aug_ + 1;

  // sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_x_);

  // timestamp of last received measurement
  previous_timestamp_ = 0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;

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

  // weight vector for sigma points
  weights_ = VectorXd(n_sigma_);

  // current NIS for radar
  NIS_radar_ = 0.0;

  // current NIS for laser
  NIS_laser_ = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_package) {

  if (!is_initialized_) {
    cout << "Unscented Kalman Filter Initialization " << endl;
    previous_timestamp_ = measurement_package.timestamp_;

    if (measurement_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      cout << "Initial measurement is RADAR" << endl;

      float rho = measurement_package.raw_measurements_[0];
      float phi = measurement_package.raw_measurements_[1];
      float rate = measurement_package.raw_measurements_[2];
      x_ << rho*cos(phi), rho*sin(phi), rate, phi, 0;

      // Start with a uniform uncertainty over the covariance
      P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    }
    else if (measurement_package.sensor_type_ == MeasurementPackage::LASER)
    {
      cout << "Initial measurement is LASER" << endl;

      float px = measurement_package.raw_measurements_[0];
      float py = measurement_package.raw_measurements_[1];
      
      // If px or py are close to zero then set them to some small positive value
      px = (fabs(px) < 1e-6) ? px : 1e-6;
      py = (fabs(py) < 1e-6) ? py : 1e-6;
      x_ << px, py, 0, 0, 0;

      // Start with a uniform uncertainty over the covariance
      P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    }

    cout << "initial state: " << x_ << endl;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  //compute the time elapsed between the current and previous measurements
  double delta_t = (measurement_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_package.timestamp_;

  // Helpful suggestion from
  // https://discussions.udacity.com/t/numerical-instability-of-the-implementation/230449/14
  // If the time difference between measurements is large then it's better to
  // make smaller incremental predictions to avoid numerical instability.
  while (delta_t > 0.1)
  {
    const double dt = 0.05;
    Predict(dt);
    delta_t -= dt;
  }
  Predict(delta_t);

  if (measurement_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    cout << "Measurement is RADAR" << endl;
    UpdateRadar(measurement_package);
  }
  else if (measurement_package.sensor_type_ == MeasurementPackage::LASER)
  {
    cout << "Measurement is LIDAR" << endl;
    UpdateLidar(measurement_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Predict(double delta_t) {

  // Create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  n_sigma_ = 2 * n_aug_ + 1;
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  //create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

  //predict sigma points
  for (int i = 0; i < n_sigma_; ++i)
  {
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
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  // set weights
  weights_ = VectorXd(n_sigma_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < n_sigma_; ++i)
  {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  //predicted state mean
  x_.fill(0.0);
  x_ = Xsig_pred_ * weights_;

  //predicted state covariance matrix
  P_.fill(0.0);
  MatrixXd x_diffs = Xsig_pred_.colwise() - x_;
  x_diffs.row(3) = x_diffs.row(3).array().unaryExpr(&Tools::ConstrainAngle);
  P_ = (x_diffs * weights_.asDiagonal()) * x_diffs.transpose(); 

  cout << "predicted x_ = " << endl;
  cout << x_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage measurement_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  int n_z = 2; // 2 values for laser measurement

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_);

  //transform sigma points into measurement space
  for (int i = 0; i < n_sigma_; ++i)
  {
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);

    // measurement model
    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);
  z_pred = Zsig * weights_; 

  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  MatrixXd z_diffs = Zsig.colwise() - z_pred;
  S = (z_diffs * weights_.asDiagonal()) * z_diffs.transpose();  

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_laspx_*std_laspx_, 0,
          0, std_laspy_*std_laspy_;
  S += R;

  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  float px = measurement_package.raw_measurements_[0];
  float py = measurement_package.raw_measurements_[1];
  z << px, py;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  MatrixXd x_diffs = Xsig_pred_.colwise() - x_;
  Tc = (x_diffs * weights_.asDiagonal()) * z_diffs.transpose();
  
  //Kalman gain K;
  MatrixXd S_inv = S.inverse();
  MatrixXd K = Tc * S_inv;

  //residual
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  
  NIS_laser_ = z_diff.transpose() * S_inv * z_diff;
  
  cout << "updated (laser) x_ = " << endl;
  cout << x_ << endl;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage measurement_package) {

  int n_z = 3; // 3 values for radar measurement

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_);

  //transform sigma points into measurement space
  for (int i = 0; i < n_sigma_; ++i)
  {
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;
	double r = sqrt(p_x*p_x + p_y*p_y);
    double phi = atan2(p_y,p_x);
    double r_dot;
    if (fabs(r) < 1e-6)
    {
       r_dot = 30; // maximum r_dot we can assume
    }
    else
    {
        r_dot = (p_x*v1 + p_y*v2 ) / r;
    }
    // measurement model
    
    Zsig(0,i) = r; 
    Zsig(1,i) = phi;
    Zsig(2,i) = r_dot;
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);
  z_pred = Zsig * weights_; 

  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  MatrixXd z_diffs = Zsig.colwise() - z_pred;
  z_diffs.row(1) = z_diffs.row(1).array().unaryExpr(&Tools::ConstrainAngle);
  S = (z_diffs * weights_.asDiagonal()) * z_diffs.transpose();
  
  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S += R;

  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  float rho = measurement_package.raw_measurements_[0];
  float phi = measurement_package.raw_measurements_[1];
  float rate = measurement_package.raw_measurements_[2];
  z << rho, phi, rate;

  //create matrix for cross correlation Tc (ideas for vectorising from slack channel)
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  MatrixXd x_diffs = Xsig_pred_.colwise() - x_;
  x_diffs.row(3) = x_diffs.row(3).array().unaryExpr(&Tools::ConstrainAngle);
  Tc = (x_diffs * weights_.asDiagonal()) * z_diffs.transpose();

  //Kalman gain K;
  MatrixXd S_inv = S.inverse();
  MatrixXd K = Tc * S_inv;

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  z_diff(1) = Tools::ConstrainAngle(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  
  NIS_radar_ = z_diff.transpose() * S_inv * z_diff;
  
  cout << "updated (radar) x_ = " << endl;
  cout << x_ << endl;
}


