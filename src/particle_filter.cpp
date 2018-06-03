/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (!initialized()) {
		default_random_engine gen;
		// Creates a normal (Gaussian) distributions for x, y, and theta.
		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);

		for (int i = 0; i < num_particles; ++i) {
			Particle p;
			p.id = i;
			p.weight = 1.0;
			p.x = dist_x(gen);
			p.y = dist_y(gen);
			p.theta = dist_theta(gen);
			particles.push_back(p);
		}
		is_initialized = true;
	}
	return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	default_random_engine gen;

	for (int i = 0; i < num_particles; ++i) {
		Particle p = particles.at(i);
		// Add sensor noise to current particle state
		// Shouldn't this be control noise instead? Seems inconsistent with the videos and my understanding of the purpose here
		// Creates a normal (Gaussian) distributions for x, y, and theta.
		normal_distribution<double> dist_x(p.x, std_pos[0]);
		normal_distribution<double> dist_y(p.y, std_pos[1]);
		normal_distribution<double> dist_theta(p.theta, std_pos[2]);
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		// Predict location based on bicylce model
		p.x += (velocity / yaw_rate)*(sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
		p.y += (velocity / yaw_rate)*(cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
		p.theta = p.theta + yaw_rate * delta_t;
		particles.at(i) = p;
	}
	return;
}

Particle ParticleFilter::dataAssociation(Particle &p, std::vector<LandmarkObs> predicted, const Map &map_landmarks) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	p.associations.clear();
	p.sense_x.clear();
	p.sense_y.clear();
	for (unsigned int i = 0; i < predicted.size(); ++i) {
		LandmarkObs pred = predicted.at(i);
		int id = 0;
		double x = 0;
		double y = 0;
		float min_distance = numeric_limits<float>::infinity();
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			Map::single_landmark_s land = map_landmarks.landmark_list.at(j);
			float distance = dist(pred.x, pred.y, land.x_f, land.y_f);
			if (distance < min_distance) {
				min_distance = distance;
				id = land.id_i;
				x = pred.x;
				y = pred.y;
			}
		}
		p.associations.push_back(id);
		p.sense_x.push_back(x);
		p.sense_y.push_back(y);
	}
	return p;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	weights.clear();
	for (int i = 0; i < num_particles; ++i) {
		Particle p = particles.at(i);
		vector<LandmarkObs> predicted; // Observations in map coordinates for Particle p;
		for (unsigned int j = 0; j < observations.size(); ++j) {
			LandmarkObs pred;
			LandmarkObs ob = observations.at(j);
			pred.x = p.x + cos(p.theta)*ob.x - sin(p.theta)*ob.y;
			pred.y = p.y + sin(p.theta)*ob.x + cos(p.theta)*ob.y;
			predicted.push_back(pred);
		}

		p = dataAssociation(p, predicted, map_landmarks);

		p.weight = 1.0;
		for (unsigned int j = 0; j < p.associations.size(); ++j) {
			int id = p.associations.at(j);
			int k = 0;
			Map::single_landmark_s landmark = map_landmarks.landmark_list.at(k);
			while (id != landmark.id_i) { landmark = map_landmarks.landmark_list.at(++k); }
			//cout << p.id << ": " << landmark.id_i << ": " << id << "\t" << p.sense_x.at(j) << "\t" << landmark.x_f <<  endl;
			p.weight *= (1.0 / (2.0*M_PI)) *
				exp(-1.0*(pow((p.sense_x.at(j) - landmark.x_f), 2) / (2.0*std_landmark[0] * std_landmark[0]) +
					pow((p.sense_y.at(j) - landmark.y_f), 2) / (2.0*std_landmark[1] * std_landmark[1])));
		}
		weights.push_back(p.weight);
		//cout << p.id << ": "<< p.weight << endl;
		particles.at(i) = p;
	}
	return;
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<> d(weights.begin(),weights.end()); // TODO: initializer list from weights
	vector<Particle> next_particles;
	for (unsigned int i = 0; i < particles.size(); ++i) {
		int index = d(gen);
		Particle p = particles.at(index);
		next_particles.push_back(p);
	}
	particles = next_particles;
	return;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
