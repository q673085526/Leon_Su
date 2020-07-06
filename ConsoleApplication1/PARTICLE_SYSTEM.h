#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "PARTICLE.h"
#include "WALL.h"
#include <vector>
#include <fstream>
#include <string>
#include <windows.h>
//#include <tr1/tuple>
//#include <map>
#include "FIELD_3D.h"
#include <cstdlib>
#include "HW1.h"


#define OUTPATH "..\\out\\"
#define h 0.0457 //0.02 //0.045

#define GAS_STIFFNESS 3.0 //20.0 // 461.5  // Nm/kg is gas constant of water vapor
#define REST_DENSITY 998.29 // kg/m^3 is rest density of water particle
#define PARTICLE_MASS 0.02 // kg
#define VISCOSITY 5 // 5.0 // 0.00089 // Ns/m^2 or Pa*s viscosity of water
#define SURFACE_TENSION 1//0.0728 // N/m 
#define SURFACE_THRESHOLD 7.065
#define KERNEL_PARTICLES 20.0

#define GRAVITY_ACCELERATION -9.80665


#define WALL_K 10000.0 // wall spring constant
#define WALL_DAMPING -0.9 // wall damping constant

#define BOX_SIZE 0.4
#define MAX_PARTICLES 3000

#define INITIAL_SCENARIO SCENARIO_DAM

#define MAXINGRID 4096

#define DRAGTIME 150
#define FIRSTFRAME 550
using namespace std;

class PARTICLE_SYSTEM {
  
public:
  PARTICLE_SYSTEM();
  ~PARTICLE_SYSTEM();

  void updateGrid();
  
  // draw to OGL
  void draw();
  
  void stepVerlet(double dt);
    
  void calculateAcceleration();
  
  void  generateDamParticleSet();
  
  
  void loadScenario(int scenario);
  
  void output(string filename1, string filename2);
  
  void readPostion(string filename);
  
  void clearPos() {
	  newPos.clear();
  }

  void add2Pos(VEC3D p, VEC3D v) {
	  newPos.push_back({ p, v });
  }
  
  vector<pair<VEC3D, VEC3D>> getPos() {
	  return newPos;
  }
  //typedef std::tr1::tuple<int,int,int> gridKey;  
  //std::map<gridKey, std::vector<PARTICLE> > grid;
  
  
  FIELD_3D* grid;
  double surfaceThreshold;
  VEC3D gravityVector;
  int dt;
private:
  // list of particles, walls, and springs being simulated
  vector<PARTICLE> _particles;
  vector<WALL>     _walls;

  //unsigned int _particleCount;
  bool _isGridVisible;
  bool _tumble;
  
  VEC3D boxSize;
  vector<pair<VEC3D, VEC3D>> newPos;
};

#endif
