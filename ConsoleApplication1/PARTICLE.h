#ifndef PARTICLE_H
#define PARTICLE_H
#define M_PI 3.1415926
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "VEC3F.h"
#include "VEC3D.h"
#include <vector>
#include <fstream>

#define BOXSIZEX 0.4  * 352/100
#define BOXSIZEY 0.4
#define BOXSIZEZ 0.4  * 288/100
#define DX  BOXSIZEX / 2 + 0.1
#define DZ  BOXSIZEZ / 2 + 0.1

using namespace std;




class PARTICLE {
  
  

  
public:
  
  static bool isSurfaceVisible;
  static bool showArrows;
  
  //static unsigned int count;
  PARTICLE();
  PARTICLE(const VEC3D& position);
  PARTICLE(const VEC3D& position, const VEC3D& velocity);
  //~PARTICLE();
  
  // draw to OGL
  void draw(VEC3F color);

  // clear all previous accumulated forces
  void clearForce() { _force *= 0; };

  // accumulate forces
  void addForce(VEC3D newForce) { _force += newForce; };
  
  void calculateAcceleration();

  // accessors
  VEC3D& position() { return _position; };
  VEC3D& velocity() { return _velocity; };
  VEC3D& acceleration() { return _acceleration; }
  VEC3D& force()    { return _force; };
  double& density()  { return _density; };
  double& pressure() { return _pressure; };
  bool& flag() { return _flag; };
  bool& exSPHflag() { return _exSPHflag; };
  double& scaleFactor() { return _scaleFactor; };
  int& id() { return _id; };
  VEC3D normal;
  
  void clearParameters();
  
  static unsigned int count;
  
private:  
  VEC3D _position;
  VEC3D _velocity;
  VEC3D _force;
  VEC3D _acceleration;
  double _density;
  double _pressure;
  bool _flag;
  int _id;
  int _sumid;
  bool _exSPHflag;   //水体粒子为true， 牵引粒子为false
  double _scaleFactor;
  GLUquadricObj* myQuadric;
    
};


#endif
