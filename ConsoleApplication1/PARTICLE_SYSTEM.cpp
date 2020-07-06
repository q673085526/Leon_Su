#include "PARTICLE_SYSTEM.h"
#define POSITTIONDIR "F:\\py_workplace\\cvOpticalFlow\\output\\gentlewaves\\xyzuvw_3\\"

extern "C" void gpu_run(FIELD_3D* field, vector<WALL> walls, int iteration, bool loadWall);


unsigned int iteration = 0;
int scenario;
bool loadWall = true;


///////////////////////////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////////////////////////
PARTICLE_SYSTEM::PARTICLE_SYSTEM() : 
_isGridVisible(false), surfaceThreshold(0.01), gravityVector(0.0,GRAVITY_ACCELERATION,0.0), grid(NULL)
{
  loadScenario(INITIAL_SCENARIO);
}

void PARTICLE_SYSTEM::loadScenario(int newScenario) {
	scenario = newScenario;
	// remove all particles
	if (grid) {
		delete grid;
	}
	_walls.clear();
	// reset params
	PARTICLE::count = 0;
	iteration = 0;
	if (scenario == SCENARIO_DAM) {
		// create long grid
		boxSize.x = BOX_SIZE * 352/100;
		boxSize.y = BOX_SIZE;
		boxSize.z = BOX_SIZE * 288/100;
		int gridXRes = (int)ceil(boxSize.x/h);
		int gridYRes = (int)ceil(boxSize.y/h);
		int gridZRes = (int)ceil(boxSize.z/h);
		grid = new FIELD_3D(gridXRes, gridYRes, gridZRes);
		// add walls 
		/*_walls.push_back(WALL(VEC3D(0,0,1), VEC3D(0,0,-boxSize.z/2.0))); // back
		_walls.push_back(WALL(VEC3D(0,0,-1), VEC3D(0,0,boxSize.z/2.0))); // front
		_walls.push_back(WALL(VEC3D(1,0,0), VEC3D(-boxSize.x/2.0,0,0)));     // left
		_walls.push_back(WALL(VEC3D(-1,0,0), VEC3D(boxSize.x/2.0,0,0)));     // right*/
		_walls.push_back(WALL(VEC3D(0,1,0), VEC3D(0,-boxSize.y/2.0,0))); // bottom
		vector<PARTICLE>& firstGridCell = (*grid)(0,0,0);
		// add particles
		string filename = POSITTIONDIR + string("951.txt");
		readPostion(filename);
		vector<pair<VEC3D, VEC3D>> pos = getPos();
		for (int i = 0; i < pos.size(); ++i) {
			PARTICLE *p = new PARTICLE(pos[i].first, pos[i].second);
			p->exSPHflag() = false;
			firstGridCell.push_back(*p);
		}
		cout << "Loaded dam scenario" << endl;
		cout << "Grid size is " << (*grid).xRes() << "x" << (*grid).yRes() << "x" << (*grid).zRes() << endl;
		cout << "Simulating " << PARTICLE::count << " particles" << endl;
	}
	updateGrid();
}

// to update the grid cells particles are located in
// should be called right after particle positions are updated
void PARTICLE_SYSTEM::updateGrid() {
    
	for (unsigned int x = 0; x < (*grid).xRes(); x++) {
		for (unsigned int y = 0; y < (*grid).yRes(); y++) {
			for (unsigned int z = 0; z < (*grid).zRes(); z++) {
        
				vector<PARTICLE>& particles = (*grid)(x,y,z);
        
				//cout << particles.size() << "p's in this grid" << endl;
                
				for (int p = 0; p < particles.size(); p++) {
          
					PARTICLE& particle = particles[p];
          
					int newGridCellX = (int)floor((particle.position().x+BOX_SIZE/2.0)/h); 
					int newGridCellY = (int)floor((particle.position().y+BOX_SIZE/2.0)/h);
					int newGridCellZ = (int)floor((particle.position().z+BOX_SIZE/2.0)/h);
          
					//cout << "particle position: " << particle.position() << endl;
					//cout << "particle cell pos: " << newGridCellX << " " << newGridCellY << " " << newGridCellZ << endl;
        
					if (newGridCellX < 0)
						newGridCellX = 0;
					else if (newGridCellX >= (*grid).xRes())
						newGridCellX = (*grid).xRes() - 1;
					if (newGridCellY < 0)
						newGridCellY = 0;
					else if (newGridCellY >= (*grid).yRes())
						newGridCellY = (*grid).yRes() - 1;
					if (newGridCellZ < 0)
						newGridCellZ = 0;
					else if (newGridCellZ >= (*grid).zRes())
						newGridCellZ = (*grid).zRes() - 1;
          
					//cout << "particle cell pos: " << newGridCellX << " " << newGridCellY << " " << newGridCellZ << endl;

          
					// check if particle has moved
          
					if ((x != newGridCellX || y != newGridCellY || z != newGridCellZ)){
            
						// move the particle to the new grid cell
						if((*grid)(newGridCellX, newGridCellY, newGridCellZ).size() < MAXINGRID)
							(*grid)(newGridCellX, newGridCellY, newGridCellZ).push_back(particle);
            
						// remove it from it's previous grid cell
            
						particles[p] = particles.back();
						particles.pop_back();
						p--; // important! make sure to redo this index, since a new particle will (probably) be there
					}
          
				}
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// OGL drawing
///////////////////////////////////////////////////////////////////////////////
void PARTICLE_SYSTEM::draw() 
{ 
  static VEC3F blackColor(0,0,0); 
  static VEC3F blueColor(0,0,1); 
  static VEC3F whiteColor(1,1,1);
  static VEC3F greyColor(0.2, 0.2, 0.2);
  static VEC3F lightGreyColor(0.8,0.8,0.8);
  //static VEC3F greenColor(34.0 / 255, 139.0 / 255, 34.0 / 255);
  static float shininess = 10.0;

  // draw the particles
  glEnable(GL_LIGHTING);
  glMaterialfv(GL_FRONT, GL_DIFFUSE, blueColor);
  glMaterialfv(GL_FRONT, GL_SPECULAR, whiteColor);
  glMaterialfv(GL_FRONT, GL_SHININESS, &shininess);
  
  //for (unsigned int x = 0; x < _particles.size(); x++)
  //  _particles[x].draw();
  
#ifdef BRUTE
  
  for (unsigned int x = 0; x < _particles.size(); x++)
    _particles[x].draw();
    
#else
  
  for (int gridCellIndex = 0; gridCellIndex < (*grid).cellCount(); gridCellIndex++) {
    
    vector<PARTICLE>& particles = (*grid).data()[gridCellIndex];
    
    for (int p = 0; p < particles.size(); p++) {
      
      PARTICLE& particle = particles[p];
	  if (particle.exSPHflag())
		  particle.draw(VEC3F(0.01, 0.25, 1.0));
	  /*else
		  particle.draw(VEC3F(0.88, 0.08, 0.88));*/
    }
    
  }
  
  glDisable(GL_LIGHTING);
  
  if (_isGridVisible) {
  
    // draw the grid
    
    glColor3fv(lightGreyColor);
    
    //double offset = -BOX_SIZE/2.0+h/2.0;
    
    for (int x = 0; x < grid->xRes(); x++) {
      for (int y = 0; y < grid->yRes(); y++) {
        for (int z = 0; z < grid->zRes(); z++) {
          glPushMatrix();
          
          glTranslated(x*h-boxSize.x/2.0+h/2.0, y*h-boxSize.y/2.0+h/2.0, z*h-boxSize.z/2.0+h/2.0);
          glutWireCube(h);
          
          glPopMatrix();
        }
      }
    }
    
  }
   

#endif

  glColor3fv(greyColor);
  
  glPopMatrix();
  glScaled(boxSize.x, boxSize.y, boxSize.z);
  glutWireCube(1.0);
  glPopMatrix();
}

///////////////////////////////////////////////////////////////////////////////
// Verlet integration
///////////////////////////////////////////////////////////////////////////////
void PARTICLE_SYSTEM::stepVerlet(double dt)                                         
{
	//每次迭代时，产生新的粒子
	DWORD start, stop;
	start = GetTickCount();

	calculateAcceleration();
	string name = to_string(iteration);
	output("..\\out\\" + name + ".txt", "..\\out\\control\\" + name + ".txt");
	/*if(iteration >= FIRSTFRAME && iteration % 50 == 0){
		string name = to_string((iteration - FIRSTFRAME) / 50 + 951);
		//string name = to_string(iteration);
		output("..\\out\\" + name + ".txt", "..\\out\\control\\" + name + ".txt");
		string filename = POSITTIONDIR + name + ".txt";
		readPostion(filename);
		vector<pair<VEC3D, VEC3D>> pos = getPos();
		vector<pair<VEC3D, VEC3D>>::iterator it = pos.begin();
		for (unsigned int gridCellIndex = 0; gridCellIndex < (*grid).cellCount(); gridCellIndex++) {
			vector<PARTICLE>& particles = (*grid).data()[gridCellIndex];
			for (unsigned int p = 0; p < particles.size(); p++) {
				PARTICLE& particle = particles[p];
				if (!particle.exSPHflag())
				{
					particle.position() = it->first;
					particle.velocity() = it->second;
					it++;
				}
			}
		}
	}*/
	if (iteration > DRAGTIME && iteration %/*8*/40 == 0 || iteration <= DRAGTIME && iteration % 9 == 0) {
		generateDamParticleSet();
	} 
	updateGrid();
	iteration++;
	stop = GetTickCount();
	printf("iteration = %d   simulate cost time: %lld ms\n", iteration, stop - start);
}


void PARTICLE_SYSTEM::generateDamParticleSet() {
	vector<PARTICLE>& firstGridCell = (*grid)(0, 0, 0);
	double ymin = -boxSize.y / 2, ymax = 0-0.05;
	for (double y = ymin; y < ymax; y += h / 2.0) {
		for (double x = -boxSize.x / 2-0.09; x < -boxSize.x / 2 + h - 0.09; x += h / 2.0) {
			for (double z = -boxSize.z / 2.0 - 0.1+h; z < boxSize.z / 2.0 + 0.1-h; z += h / 2.0 ) {
				PARTICLE p = PARTICLE(VEC3D(x, y, z));
				p.exSPHflag() = true;
				firstGridCell.push_back(p);
			}
		}
	}
	
	for (double y = ymin; y < ymax; y += h / 2.0) {
		for (double x = boxSize.x / 2 - h + 0.09; x < boxSize.x / 2+0.09; x += h / 2.0) {
			for (double z = -boxSize.z / 2.0 - 0.1+h; z < boxSize.z / 2.0 + 0.1-h; z += h / 2.0) {
				PARTICLE p = PARTICLE(VEC3D(x, y, z));
				p.exSPHflag() = true;
				firstGridCell.push_back(p);
			}
		}
	}

	for (double y = ymin; y < ymax; y += h / 2.0) {
		for (double x = -boxSize.x / 2-0.1+h; x < boxSize.x / 2+0.1-h; x += h / 2.0) {
			for (double z = -boxSize.z / 2.0 - 0.09; z < -boxSize.z / 2.0 + h - 0.09; z += h / 2.0) {
				PARTICLE p = PARTICLE(VEC3D(x, y, z));
				p.exSPHflag() = true;
				firstGridCell.push_back(p);
			}
		}
	}

	for (double y = ymin; y < ymax; y += h / 2.0) {
		for (double x = -boxSize.x / 2-0.1+h; x < boxSize.x / 2+0.1-h; x += h / 2.0) {
			for (double z = boxSize.z / 2.0 -h +0.09; z < boxSize.z / 2.0+0.09; z += h / 2.0) {
				PARTICLE p = PARTICLE(VEC3D(x, y, z));
				p.exSPHflag() = true;
				firstGridCell.push_back(p);
			}
		}
	}
}

/*
 Calculate the acceleration of each particle using a grid optimized approach.
 For each particle, only particles in the same grid cell and the (26) neighboring grid cells must be considered,
 since any particle beyond a grid cell distance away contributes no force.
*/
void PARTICLE_SYSTEM::calculateAcceleration() {
  
  ///////////////////
  // STEP 1: UPDATE DENSITY & PRESSURE OF EACH PARTICLE
	bool gpu = true;
	
	if (gpu) {
		gpu_run(grid, _walls, iteration, loadWall);
		if (loadWall)
			loadWall = false;
	}
}


void PARTICLE_SYSTEM::output(string filename1, string filename2) {
	ofstream out1(filename1);
	ofstream out2(filename2);
	for (int gridCellIndex = 0; gridCellIndex < (*grid).cellCount(); gridCellIndex++) {
		vector<PARTICLE>& particles = (*grid).data()[gridCellIndex];
		for (int p = 0; p < particles.size(); p++) {
			PARTICLE& particle = particles[p];
			if(particle.exSPHflag())
				out1 << particle.position().x << ";" << particle.position().y << ";" << particle.position().z << ";" << particle.velocity().x << ";" << particle.velocity().y << ";" << particle.velocity().z << endl;
			//if(particle.exSPHflag() && abs(particle.position().x) < BOXSIZEX/2 && abs(particle.position().z) < BOXSIZEZ/2)
				//out1 << particle.position().x << ";" << particle.position().y << ";" << particle.position().z << ";" << particle.velocity().x << ";" << particle.velocity().y << ";" << particle.velocity().z << endl;
			//if(!particle.exSPHflag())
				//out2 << particle.position().x << ";" << particle.position().y << ";" << particle.position().z << ";" << particle.velocity().x << ";" << particle.velocity().y << ";" << particle.velocity().z << endl;
		}
	}
	out1.close();
	out2.close();
}

void PARTICLE_SYSTEM::readPostion(string filename) {
	clearPos();
	ifstream f(filename);
	if (!f.is_open()) {
		cout << "file read error at " << filename << endl;
		//exit(1);
	}
	cout << "read pos from " << filename << endl;
	float x, y, z, u, v, w;
	char c;
	int i = 0;
	while (!f.eof()) {
		f >> z >> x >> y >> u >> v >> w;
		//f >> x >> c >> y >> c >> z >> c >> u >> c >> v >> c >> w;
		//add2Pos(VEC3D(x * 2, y * 1.5 - 0.025, z * 2), VEC3D(u * 2, w * 2, v * 2));
		/* 
			通过对牵引粒子的位置进行插值，由单层粒子牵引转为体牵引
		*/
	/*	if(i%2 == 0)
		{*/
		y -= 0.01;
		int k = 3;
		while (k > 0) {
			add2Pos(VEC3D(x * 2, y, z * 2), VEC3D(u * 2, w, v * 2));
			y -= h;
			--k;
		}
	/*	}
		++i;*/
	}
}