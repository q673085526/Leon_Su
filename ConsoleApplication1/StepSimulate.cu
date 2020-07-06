#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PARTICLE_SYSTEM.h"
#include <stdio.h>
#include <stdlib.h>
#define THREAD 1024
#define attractFactor  0.070f / h//0.045f / h;
#define velocityFactor  0.005f / h;
#define pa d_data[idx][i * THREAD + pidx]
#define pb d_data[nidx][j]
#pragma unroll
#define BOXSIZE 0.4
//#define dt 0.01


typedef struct vec3d {
	float x;
	float y;
	float z;
}vec3d;
__device__ vec3d newVec3d(float x, float y, float z) {
	vec3d temp;
	temp.x = x; temp.y = y; temp.z = z;
	return temp;
}
typedef struct particle {
	vec3d position;
	vec3d velocity;
	vec3d force;
	vec3d acceleration;
	//vec3d normal;
	float density;
	float scaleFactor;
	float pressure;
	bool flag;
	bool exSPHflag;
	//bool exSPHbottomflag;
	//bool exSPHsurfaceflag;
	//float exdensity;
	//float expressure;
	//vec3d exacceleration;
	int id;
	int sumid;
}particle;
typedef struct wall {
	vec3d normal;
	vec3d point;
}wall;
//取出grid(x, y, z)中的粒子

inline __host__ __device__ int grid_index(int x, int y, int z, int xRes, int yRes, int zRes) {
	return x + y * xRes + z * xRes*yRes; // return max = xRes*yRes*zRes - 1
}
//核函数
__device__ float Wpoly6(float radiusSquared)
{
	const float coefficient = 315.0f / (64.0f*M_PI*pow(h, 9));
	const float hSquared = h * h;
	return coefficient * pow(hSquared - radiusSquared, 3);
}
__device__ void Wpoly6Gradient(vec3d& diffPosition, float radiusSquared, vec3d& gradient)
{
	const float coefficient = -945.0f / (32.0f*M_PI*pow(h, 9));
	const float hSquared = h * h;
	//gradient = coefficient * pow(hSquared - radiusSquared, 2) * diffPosition;
	gradient.x = coefficient * pow(hSquared - radiusSquared, 2) * diffPosition.x;
	gradient.y = coefficient * pow(hSquared - radiusSquared, 2) * diffPosition.y;
	gradient.z = coefficient * pow(hSquared - radiusSquared, 2) * diffPosition.z;
}
__device__ float Wpoly6Laplacian(float radiusSquared)
{
	const float coefficient = -945.0f / (32.0f*M_PI*pow(h, 9));
	const float hSquared = h * h;
	return coefficient * (hSquared - radiusSquared) * (3.0*hSquared - 7.0*radiusSquared);
}
__device__ void WspikyGradient(vec3d& diffPosition, float radiusSquared, vec3d& gradient)
{
	const float coefficient = -45.0f / (M_PI*pow(h, 6));
	float radius = sqrt(radiusSquared);
	gradient.x = coefficient * pow(h - radius, 2) * diffPosition.x / radius;
	gradient.y = coefficient * pow(h - radius, 2) * diffPosition.y / radius;
	gradient.z = coefficient * pow(h - radius, 2) * diffPosition.z / radius;
}
__device__ float WviscosityLaplacian(float radiusSquared)
{
	const float coefficient = 45.0f / (M_PI*pow(h, 6));
	float radius = sqrt(radiusSquared);
	return coefficient * (h - radius);
}
__device__ void collisionForce_z(particle &p, wall *d_wall, int wSize) {
	int i = 0;
	for (i = 0; i < wSize; i++)
	{
		wall wall = d_wall[i];
		float d = (wall.point.x - p.position.x)*wall.normal.x +
			(wall.point.y - p.position.y)*wall.normal.y +
			(wall.point.z - p.position.z)*wall.normal.z + 0.015f; // d为穿透深度
		if (d > 0.0f)
		{
			p.acceleration.x += WALL_K * wall.normal.x * d;
			p.acceleration.y += WALL_K * wall.normal.y * d;
			p.acceleration.z += WALL_K * wall.normal.z * d;
			p.acceleration.x += (WALL_DAMPING * p.velocity.x *wall.normal.x +
				WALL_DAMPING * p.velocity.y *wall.normal.y +
				WALL_DAMPING * p.velocity.z *wall.normal.z) * wall.normal.x;
			p.acceleration.y += (WALL_DAMPING * p.velocity.x *wall.normal.x +
				WALL_DAMPING * p.velocity.y *wall.normal.y +
				WALL_DAMPING * p.velocity.z *wall.normal.z) * wall.normal.y;
			p.acceleration.z += (WALL_DAMPING * p.velocity.x *wall.normal.x +
				WALL_DAMPING * p.velocity.y *wall.normal.y +
				WALL_DAMPING * p.velocity.z *wall.normal.z) * wall.normal.z;
		}
	}
}
int xRes, yRes, zRes;
particle **h_data;   // host field
particle **d_data;   //device field
particle **new_data; //更新后的网格
wall *h_wall, *d_wall;
int *h_size, *d_size;
int *new_size; //更新后的size
inline void CHECK(cudaError err)//错误处理函数
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA Runtime API error: %s.\n", cudaGetErrorString(err));
		return;
	}
}
//参数：d_data设备内存上的粒子，分为若干个grid，d_size
__global__ void caculateDensity(particle** d_data, int* d_size, int xRes, int yRes, int zRes) {
	int idx = grid_index((int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, xRes, yRes, zRes);
	// 访问grid(blockIdx.x, blockIdx.y, blockIdx.z)
	int pidx = (int)threadIdx.x;
	int i;
	if (idx < xRes*yRes*zRes) {
		for (i = 0; i * THREAD + pidx < d_size[idx]; ++i) {
			if (d_data[idx][i * THREAD + pidx].exSPHflag) { //对于水体粒子求密度
				d_data[idx][i * THREAD + pidx].density = 0;
				//caculate density
				int x, y, z;
				int j;
				int nsize, nidx;
				for (x = (int)blockIdx.x - 1; x <= (int)blockIdx.x + 1; ++x) {
					if (x < 0) continue;
					if (x >= xRes) break;
					for (y = (int)blockIdx.y - 1; y <= (int)blockIdx.y + 1; ++y) {
						if (y < 0) continue;
						if (y >= yRes) break;
						for (z = (int)blockIdx.z - 1; z <= (int)blockIdx.z + 1; ++z) {
							if (z < 0) continue;
							if (z >= zRes) break;
							nidx = grid_index(x, y, z, xRes, yRes, zRes);
							if (nidx < xRes*yRes*zRes) {
								nsize = d_size[nidx];
								for (j = 0; j < nsize; ++j) {
									if (d_data[nidx][j].exSPHflag) {	//周围的水体粒子才对密度值有贡献，牵引粒子不影响密度
										vec3d diffPosition;
										diffPosition.x = d_data[idx][i * THREAD + pidx].position.x - d_data[nidx][j].position.x;
										diffPosition.y = d_data[idx][i * THREAD + pidx].position.y - d_data[nidx][j].position.y;
										diffPosition.z = d_data[idx][i * THREAD + pidx].position.z - d_data[nidx][j].position.z;
										float radiusSquared = diffPosition.x * diffPosition.x + diffPosition.y * diffPosition.y + diffPosition.z * diffPosition.z;
										if (radiusSquared <= h * h)
											d_data[idx][i * THREAD + pidx].density += Wpoly6(radiusSquared);
									}
								}
							}
						}
					}
				}
				d_data[idx][i * THREAD + pidx].density *= PARTICLE_MASS;
				d_data[idx][i * THREAD + pidx].pressure = GAS_STIFFNESS * (d_data[idx][i * THREAD + pidx].density - REST_DENSITY);
			}
		}
	}
}
__global__ void caculateScaleFactor(particle** d_data, int* d_size, int xRes, int yRes, int zRes) {
	int idx = grid_index((int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, xRes, yRes, zRes);
	// 访问grid(blockIdx.x, blockIdx.y, blockIdx.z)
	int pidx = (int)threadIdx.x;
	int i;
	if (idx < xRes*yRes*zRes) {
		for (i = 0; i * THREAD + pidx < d_size[idx]; ++i) {
			if (!d_data[idx][i * THREAD + pidx].exSPHflag) {	//对于牵引粒子计算牵引力系数
				d_data[idx][i * THREAD + pidx].scaleFactor = 0;
				//caculate density
				int x, y, z;
				int j;
				int nsize, nidx;
				for (x = (int)blockIdx.x - 1; x <= (int)blockIdx.x + 1; ++x) {
					if (x < 0) continue;
					if (x >= xRes) break;
					for (y = (int)blockIdx.y - 1; y <= (int)blockIdx.y + 1; ++y) {
						if (y < 0) continue;
						if (y >= yRes) break;
						for (z = (int)blockIdx.z - 1; z <= (int)blockIdx.z + 1; ++z) {
							if (z < 0) continue;
							if (z >= zRes) break;
							nidx = grid_index(x, y, z, xRes, yRes, zRes);
							if (nidx < xRes*yRes*zRes) {
								nsize = d_size[nidx];
								for (j = 0; j < nsize; ++j) {
									if (d_data[nidx][j].exSPHflag) {	
										vec3d diffPosition;
										diffPosition.x = d_data[idx][i * THREAD + pidx].position.x - d_data[nidx][j].position.x;
										diffPosition.y = d_data[idx][i * THREAD + pidx].position.y - d_data[nidx][j].position.y;
										diffPosition.z = d_data[idx][i * THREAD + pidx].position.z - d_data[nidx][j].position.z;
										float radiusSquared = diffPosition.x * diffPosition.x + diffPosition.y * diffPosition.y + diffPosition.z * diffPosition.z;
										if (radiusSquared <= h * h)
											d_data[idx][i * THREAD + pidx].scaleFactor += 1.0 / d_data[nidx][j].density * Wpoly6(radiusSquared);
									}
								}
							}
						}
					}
				}
				d_data[idx][i * THREAD + pidx].scaleFactor *= PARTICLE_MASS;
				d_data[idx][i * THREAD + pidx].scaleFactor = (1.0 < (d_data[idx][i * THREAD + pidx].scaleFactor)) ? 0.0 : (1.0 - d_data[idx][i * THREAD + pidx].scaleFactor);
			}
		}
	}
}
__global__ void caculateAllForce(particle** d_data, int* d_size, int xRes, int yRes, int zRes, int iter) {
	vec3d f_pressure,
		f_viscosity;
	vec3d	f_gravity, f_surface,
		colorFieldNormal,
		f_attract,
		f_velocity;
	float sqrtc;
	int idx = grid_index((int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, xRes, yRes, zRes);
	// 访问grid(blockIdx.x, blockIdx.y, blockIdx.z)
	int pidx = (int)threadIdx.x;
	int i;
	if (idx < xRes*yRes*zRes) {
		for (i = 0; i * THREAD + pidx < d_size[idx]; ++i) {
			if (d_data[idx][i * THREAD + pidx].exSPHflag) {		//对于水体粒子，计算其受力情况
				f_pressure = newVec3d(0, 0, 0),
					f_viscosity = newVec3d(0, 0, 0),
					f_surface = newVec3d(0, 0, 0),
					f_gravity = newVec3d(0.0, GRAVITY_ACCELERATION * d_data[idx][i * THREAD + pidx].density, 0.0),
					colorFieldNormal = newVec3d(0, 0, 0),
					f_attract = newVec3d(0, 0, 0),
					f_velocity = newVec3d(0, 0, 0);
				float colorFieldLaplacian = 0.0;
				//caculate density
				int x, y, z;
				int j;
				int nsize, nidx;
				for (x = (int)blockIdx.x - 1; x <= (int)blockIdx.x + 1; ++x) {
					if (x < 0) continue;
					if (x >= xRes) break;
					for (y = (int)blockIdx.y - 1; y <= (int)blockIdx.y + 1; ++y) {
						if (y < 0) continue;
						if (y >= yRes) break;
						for (z = (int)blockIdx.z - 1; z <= (int)blockIdx.z + 1; ++z) {
							if (z < 0) continue;
							if (z >= zRes) break;
							nidx = grid_index(x, y, z, xRes, yRes, zRes);
							if (nidx < xRes*yRes*zRes) {
								for (j = 0; j < d_size[nidx]; ++j) {
									vec3d diffPosition;
									diffPosition.x = d_data[idx][i * THREAD + pidx].position.x - d_data[nidx][j].position.x;
									diffPosition.y = d_data[idx][i * THREAD + pidx].position.y - d_data[nidx][j].position.y;
									diffPosition.z = d_data[idx][i * THREAD + pidx].position.z - d_data[nidx][j].position.z;
									float radiusSquared = diffPosition.x * diffPosition.x + diffPosition.y * diffPosition.y + diffPosition.z * diffPosition.z;
									if (radiusSquared <= h * h && d_data[nidx][j].exSPHflag) { //水体粒子提供的压力和粘性力
										vec3d poly6Gradient, spikyGradient;
										Wpoly6Gradient(diffPosition, radiusSquared, poly6Gradient);
										WspikyGradient(diffPosition, radiusSquared, spikyGradient);
										if (d_data[idx][i * THREAD + pidx].id != d_data[nidx][j].id) {
											f_pressure.x += (d_data[idx][i * THREAD + pidx].pressure / pow(d_data[idx][i * THREAD + pidx].density, 2) + d_data[nidx][j].pressure / pow(d_data[nidx][j].density, 2))*spikyGradient.x;
											f_pressure.y += (d_data[idx][i * THREAD + pidx].pressure / pow(d_data[idx][i * THREAD + pidx].density, 2) + d_data[nidx][j].pressure / pow(d_data[nidx][j].density, 2))*spikyGradient.y;
											f_pressure.z += (d_data[idx][i * THREAD + pidx].pressure / pow(d_data[idx][i * THREAD + pidx].density, 2) + d_data[nidx][j].pressure / pow(d_data[nidx][j].density, 2))*spikyGradient.z;
											f_viscosity.x += (d_data[nidx][j].velocity.x - d_data[idx][i * THREAD + pidx].velocity.x) * WviscosityLaplacian(radiusSquared) / d_data[nidx][j].density;
											f_viscosity.y += (d_data[nidx][j].velocity.y - d_data[idx][i * THREAD + pidx].velocity.y) * WviscosityLaplacian(radiusSquared) / d_data[nidx][j].density;
											f_viscosity.z += (d_data[nidx][j].velocity.z - d_data[idx][i * THREAD + pidx].velocity.z) * WviscosityLaplacian(radiusSquared) / d_data[nidx][j].density;
										}
										colorFieldNormal.x += poly6Gradient.x / d_data[nidx][j].density;
										colorFieldNormal.y += poly6Gradient.y / d_data[nidx][j].density;
										colorFieldNormal.z += poly6Gradient.z / d_data[nidx][j].density;
										colorFieldLaplacian += Wpoly6Laplacian(radiusSquared) / d_data[nidx][j].density;
									}
									diffPosition.x = -d_data[idx][i * THREAD + pidx].position.x + d_data[nidx][j].position.x;
									diffPosition.y = -d_data[idx][i * THREAD + pidx].position.y + d_data[nidx][j].position.y;
									diffPosition.z = -d_data[idx][i * THREAD + pidx].position.z + d_data[nidx][j].position.z;
									if(iter > DRAGTIME && radiusSquared <= h * h && !d_data[nidx][j].exSPHflag){ //牵引粒子计算位置牵引力和速度牵引力
										f_attract.x += d_data[nidx][j].scaleFactor*diffPosition.x / sqrt(radiusSquared) * Wpoly6(radiusSquared);
										f_attract.y += d_data[nidx][j].scaleFactor*diffPosition.y / sqrt(radiusSquared) * Wpoly6(radiusSquared);
										f_attract.z += d_data[nidx][j].scaleFactor*diffPosition.z / sqrt(radiusSquared) * Wpoly6(radiusSquared);
										f_velocity.x += (d_data[nidx][j].velocity.x - d_data[idx][i * THREAD + pidx].velocity.x)*Wpoly6(radiusSquared);
										f_velocity.y += (d_data[nidx][j].velocity.y - d_data[idx][i * THREAD + pidx].velocity.y)*Wpoly6(radiusSquared);
										f_velocity.z += (d_data[nidx][j].velocity.z - d_data[idx][i * THREAD + pidx].velocity.z)*Wpoly6(radiusSquared);
									}
								}
							}
						}
					}
				}
				f_pressure.x *= -PARTICLE_MASS * d_data[idx][i * THREAD + pidx].density;
				f_pressure.y *= -PARTICLE_MASS * d_data[idx][i * THREAD + pidx].density;
				f_pressure.z *= -PARTICLE_MASS * d_data[idx][i * THREAD + pidx].density;
				f_viscosity.x = VISCOSITY * PARTICLE_MASS * f_viscosity.x;
				f_viscosity.y = VISCOSITY * PARTICLE_MASS * f_viscosity.y;
				f_viscosity.z = VISCOSITY * PARTICLE_MASS * f_viscosity.z;

				f_attract.x *= attractFactor;
				f_attract.y *= attractFactor;
				f_attract.z *= attractFactor;
				f_velocity.x *= velocityFactor;
				f_velocity.y *= velocityFactor;
				f_velocity.z *= velocityFactor;
				colorFieldNormal.x *= PARTICLE_MASS;
				colorFieldNormal.y *= PARTICLE_MASS;
				colorFieldNormal.z *= PARTICLE_MASS;
				/*d_data[idx][i * THREAD + pidx].normal.x = -1.0 * colorFieldNormal.x;
				d_data[idx][i * THREAD + pidx].normal.y = -1.0 * colorFieldNormal.y;
				d_data[idx][i * THREAD + pidx].normal.z = -1.0 * colorFieldNormal.z;*/
				colorFieldLaplacian *= PARTICLE_MASS;
				//int sqrtc;
				sqrtc = sqrt(colorFieldNormal.x * colorFieldNormal.x + colorFieldNormal.y * colorFieldNormal.y + colorFieldNormal.z * colorFieldNormal.z);
				// surface tension force
				if (sqrtc > SURFACE_THRESHOLD) {
					d_data[idx][i * THREAD + pidx].flag = true;
					f_surface.x = -SURFACE_TENSION * colorFieldNormal.x  * colorFieldLaplacian / sqrtc;
					f_surface.y = -SURFACE_TENSION * colorFieldNormal.y  * colorFieldLaplacian / sqrtc;
					f_surface.z = -SURFACE_TENSION * colorFieldNormal.z  * colorFieldLaplacian / sqrtc;
				}
				else {
					d_data[idx][i * THREAD + pidx].flag = false;
				}
				// ADD IN SPH FORCES
				d_data[idx][i * THREAD + pidx].acceleration.x = (f_pressure.x + f_viscosity.x + f_surface.x + f_gravity.x + f_attract.x + f_velocity.x) / d_data[idx][i * THREAD + pidx].density;
				d_data[idx][i * THREAD + pidx].acceleration.y = (f_pressure.y + f_viscosity.y + f_surface.y + f_gravity.y + f_attract.y + f_velocity.y) / d_data[idx][i * THREAD + pidx].density;
				d_data[idx][i * THREAD + pidx].acceleration.z = (f_pressure.z + f_viscosity.z + f_surface.z + f_gravity.z + f_attract.z + f_velocity.z) / d_data[idx][i * THREAD + pidx].density;
			}
		}
	}
}
__global__ void caculateCollision(particle** d_data, int* d_size, int xRes, int yRes, int zRes, wall* d_wall, int wallSize) {
	int idx = grid_index((int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, xRes, yRes, zRes);
	// 访问grid(blockIdx.x, blockIdx.y, blockIdx.z)
	int pidx = (int)threadIdx.x;
	int i;
	if (idx < xRes*yRes*zRes) {
		for (i = 0; i * THREAD + pidx < d_size[idx]; ++i) {
			if (d_data[idx][i * THREAD + pidx].exSPHflag) {
				collisionForce_z(d_data[idx][i * THREAD + pidx], d_wall, wallSize);
			}
		}
	}
}

__global__ void updateGrid(particle** d_data, int* d_size, int xRes, int yRes, int zRes, particle** new_data, int* new_lock, int* new_size) {
	//将原始grid中的每个粒子放入新的grid中，新的particle in gird关系保存在new_data中
	int idx = grid_index((int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, xRes, yRes, zRes);
	// 访问grid(blockIdx.x, blockIdx.y, blockIdx.z)
	int pidx = (int)threadIdx.x;
	// gird中的第（pidx + n*thread) 个粒子
	int i;
	if (idx < xRes*yRes*zRes) {
		for (i = 0; i * THREAD + pidx < d_size[idx]; ++i) {
			//计算该粒子所在的网格
			int x = (int)floor((d_data[idx][i * THREAD + pidx].position.x + BOXSIZE / 2.0) / h);
			int y = (int)floor((d_data[idx][i * THREAD + pidx].position.y + BOXSIZE / 2.0) / h);
			int z = (int)floor((d_data[idx][i * THREAD + pidx].position.z + BOXSIZE / 2.0) / h);
			if (x < 0) x = 0;
			else if (x >= xRes) x = xRes - 1;
			if (y < 0) y = 0;
			else if (y >= yRes) y = yRes - 1;
			if (z < 0) z = 0;
			else if (z >= zRes) z = zRes - 1;
			//将该粒子加入网格
			int newidx = grid_index(x, y, z, xRes, yRes, zRes);
			//如果该网格没有粒子正在加入即lock为false，则标记lock为true，开始加入粒子
			//if (atomicExch(&new_lock[newidx], 1)==0) {
			while (new_lock[newidx] != 0) {
				if(grid_index(x, y, z, xRes, yRes, zRes) == 5)
					printf("%d synchronized!!\n", (int)threadIdx.x);
			}
			atomicExch(&new_lock[newidx], 1);
			//d_data[idx][i * THREAD + pidx];
			//memcpy(&new_data[newidx][new_size[newidx]], &d_data[idx][i * THREAD + pidx], sizeof(particle));
			
			/*if (grid_index(x, y, z, xRes, yRes, zRes) == 5)
				printf("particles number in %d is %d\n", grid_index(x, y, z, xRes, yRes, zRes), new_size[newidx]);*/
			atomicAdd(&new_size[newidx], 1);
			atomicExch(&new_lock[newidx], 0);
		}
	}
}

void updataGridOnHost(particle** h_data, int* h_size, int xRes, int yRes, int zRes, particle** &new_data,  int* new_size) {
	for (int x = 0; x < xRes; ++x) {
		for (int y = 0; y < yRes; ++y) {
			for (int z = 0; z < zRes; ++z) {
				int idx = grid_index(x, y, z, xRes, yRes, zRes);
				for (int i = 0; i < h_size[idx]; ++i) {
					particle particle = h_data[idx][i];
					int newGridCellX = (int)floor((particle.position.x + BOXSIZEX / 2.0) / h);
					int newGridCellY = (int)floor((particle.position.y + BOXSIZEY / 2.0) / h);
					int newGridCellZ = (int)floor((particle.position.z + BOXSIZEZ / 2.0) / h);
					if (newGridCellX < 0)
						newGridCellX = 0;
					else if (newGridCellX >= xRes)
						newGridCellX = xRes - 1;
					if (newGridCellY < 0)
						newGridCellY = 0;
					else if (newGridCellY >= yRes)
						newGridCellY = yRes - 1;
					if (newGridCellZ < 0)
						newGridCellZ = 0;
					else if (newGridCellZ >= zRes)
						newGridCellZ = zRes - 1;
					int newidx = grid_index(newGridCellX, newGridCellY, newGridCellZ, xRes, yRes, zRes);
					/*new_data[newidx][new_size[newidx]].acceleration.x = particle.acceleration.x;
					new_data[newidx][new_size[newidx]].acceleration.y = particle.acceleration.y;
					new_data[newidx][new_size[newidx]].acceleration.z = particle.acceleration.z;
					new_data[newidx][new_size[newidx]].position.x = particle.position.x;
					new_data[newidx][new_size[newidx]].position.y = particle.position.y;
					new_data[newidx][new_size[newidx]].position.z = particle.position.z;
					new_data[newidx][new_size[newidx]].velocity.x = particle.velocity.x;
					new_data[newidx][new_size[newidx]].velocity.y = particle.velocity.y;
					new_data[newidx][new_size[newidx]].velocity.z = particle.velocity.z;
					new_data[newidx][new_size[newidx]].exSPHflag = particle.exSPHflag;
					new_data[newidx][new_size[newidx]].id = particle.id;*/
					//new_data[newidx][new_size[newidx]] = particle;
					//memcpy(&new_data[newidx][new_size[newidx]], &h_data[idx][i], sizeof(particle));
					new_size[newidx] += 1;
					//printf("(%d, %d, %d) newidx = %d size = %d\n ", newGridCellX, newGridCellY, newGridCellZ, newidx, new_size[newidx]);
				}
			}
		}
	}
}

__global__ void updatePosition(particle** d_data, int* d_size, int xRes, int yRes, int zRes, int iter) {
	int idx = grid_index((int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, xRes, yRes, zRes);
	// 访问grid(blockIdx.x, blockIdx.y, blockIdx.z)
	int pidx = (int)threadIdx.x;
	int i;
	double dt = iter >= DRAGTIME ? 0.002 : 0.01;
	if (idx < xRes*yRes*zRes) {
		for (i = 0; i * THREAD + pidx < d_size[idx]; ++i) {
			if (d_data[idx][i * THREAD + pidx].exSPHflag) {
				d_data[idx][i * THREAD + pidx].position.x += d_data[idx][i * THREAD + pidx].velocity.x * dt + d_data[idx][i * THREAD + pidx].acceleration.x * dt * dt;
				d_data[idx][i * THREAD + pidx].position.y += d_data[idx][i * THREAD + pidx].velocity.y * dt + d_data[idx][i * THREAD + pidx].acceleration.y * dt * dt;
				d_data[idx][i * THREAD + pidx].position.z += d_data[idx][i * THREAD + pidx].velocity.z * dt + d_data[idx][i * THREAD + pidx].acceleration.z * dt * dt;
				d_data[idx][i * THREAD + pidx].velocity.x += d_data[idx][i * THREAD + pidx].acceleration.x * dt;
				d_data[idx][i * THREAD + pidx].velocity.y += d_data[idx][i * THREAD + pidx].acceleration.y * dt;
				d_data[idx][i * THREAD + pidx].velocity.z += d_data[idx][i * THREAD + pidx].acceleration.z * dt;
			}
		}
	}
}

void readPosition(FIELD_3D* field){
	for (int x = 0; x < xRes; x++) {
		for (int y = 0; y < yRes; y++) {
			for (int z = 0; z < zRes; z++) {
				vector<PARTICLE>& par = (*field)(x, y, z);
				h_size[grid_index(x, y, z, xRes, yRes, zRes)] = par.size();
				h_data[grid_index(x, y, z, xRes, yRes, zRes)] = (particle*)malloc(MAXINGRID * sizeof(particle));
				memset(h_data[grid_index(x, y, z, xRes, yRes, zRes)], 0, sizeof(particle) * MAXINGRID);
				//分配GPU空间存储粒子
				particle* d_temp;
				CHECK(cudaMalloc((void**)&d_temp, MAXINGRID * sizeof(particle)));
				for (int p = 0; p < h_size[grid_index(x, y, z, xRes, yRes, zRes)]; ++p) {
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].position.x = par[p].position().x;
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].position.y = par[p].position().y;
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].position.z = par[p].position().z;
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].id = par[p].id();
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].exSPHflag = true;
				}
				CHECK(cudaMemcpy(d_temp, h_data[grid_index(x, y, z, xRes, yRes, zRes)], sizeof(particle) * MAXINGRID, cudaMemcpyHostToDevice));
				free(h_data[grid_index(x, y, z, xRes, yRes, zRes)]);
				h_data[grid_index(x, y, z, xRes, yRes, zRes)] = d_temp;
			}//grid.x
		}//grid.y
	}//grid.z
	CHECK(cudaMemcpy(d_size, h_size, sizeof(int) * xRes*yRes*zRes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_data, h_data, sizeof(particle*) * xRes*yRes*zRes, cudaMemcpyHostToDevice));
}
extern "C" void gpu_run(FIELD_3D* field, vector<WALL> walls, int iter, bool loadwall) {
	//size用于记录每个gird中粒子的数量
	if (xRes != (*field).xRes() || yRes != (*field).yRes() || zRes != (*field).zRes()) {
		xRes = (*field).xRes();
		yRes = (*field).yRes();
		zRes = (*field).zRes();
		free(h_size);
		free(h_data);
		cudaFree(d_size);
		cudaFree(d_data);
		h_size = (int*)malloc(sizeof(int) * xRes * yRes * zRes);
		h_data = (particle**)malloc(sizeof(particle*) * xRes * yRes * zRes);
		CHECK(cudaMalloc((void**)&d_data, sizeof(particle*) * xRes * yRes * zRes));
		CHECK(cudaMalloc((void**)&d_size, sizeof(int)  * xRes * yRes * zRes));
		printf("cuda remalloc data!!\n");
		//readPosition(field);
	}
	for (int x = 0; x < xRes; x++) {
		for (int y = 0; y < yRes; y++) {
			for (int z = 0; z < zRes; z++) {
				vector<PARTICLE>& par = (*field)(x, y, z);
				h_size[grid_index(x, y, z, xRes, yRes, zRes)] = par.size();
				h_data[grid_index(x, y, z, xRes, yRes, zRes)] = (particle*)malloc(MAXINGRID * sizeof(particle));
				//memset(h_data[grid_index(x, y, z, xRes, yRes, zRes)], 0, sizeof(particle) * MAXINGRID);
				//分配GPU空间存储粒子
				particle* d_temp;
				CHECK(cudaMalloc((void**)&d_temp, MAXINGRID * sizeof(particle)));
				for (int p = 0; p < h_size[grid_index(x, y, z, xRes, yRes, zRes)] && p < MAXINGRID; ++p) {
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].position.x = par[p].position().x;
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].position.y = par[p].position().y;
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].position.z = par[p].position().z;
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].velocity.x = par[p].velocity().x;
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].velocity.y = par[p].velocity().y;
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].velocity.z = par[p].velocity().z;
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].acceleration.x = par[p].acceleration().x;
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].acceleration.y = par[p].acceleration().y;
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].acceleration.z = par[p].acceleration().z;
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].id = par[p].id();
					h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].exSPHflag = par[p].exSPHflag();
				}
				CHECK(cudaMemcpy(d_temp, h_data[grid_index(x, y, z, xRes, yRes, zRes)], sizeof(particle) * MAXINGRID, cudaMemcpyHostToDevice));
				free(h_data[grid_index(x, y, z, xRes, yRes, zRes)]);
				//cout << "grid: " << x << "," << y << "," << z << endl;
				h_data[grid_index(x, y, z, xRes, yRes, zRes)] = NULL;
				h_data[grid_index(x, y, z, xRes, yRes, zRes)] = d_temp;
			}//grid.x
		}//grid.y
	}//grid.z
	CHECK(cudaMemcpy(d_size, h_size, sizeof(int) * xRes*yRes*zRes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_data, h_data, sizeof(particle*) * xRes*yRes*zRes, cudaMemcpyHostToDevice));
	//将wall数据存储到主机内存中
	if (loadwall) {
		h_wall = (wall*)malloc(sizeof(wall) * walls.size());
		CHECK(cudaMalloc((void**)&d_wall, sizeof(wall) * walls.size()));
		for (int i = 0; i < walls.size(); ++i) {
			h_wall[i].normal.x = walls[i].normal().x;
			h_wall[i].normal.y = walls[i].normal().y;
			h_wall[i].normal.z = walls[i].normal().z;
			h_wall[i].point.x = walls[i].point().x;
			h_wall[i].point.y = walls[i].point().y;
			h_wall[i].point.z = walls[i].point().z;
		}
		CHECK(cudaMemcpy(d_wall, h_wall, sizeof(wall) * walls.size(), cudaMemcpyHostToDevice));
		free(h_wall);
	}
	//printf("class >>> struct  copy time: %lld ms\n", stop - start);
	dim3 block(xRes, yRes, zRes);
	dim3 thread(THREAD);
	caculateDensity << <block, thread >> > (d_data, d_size, xRes, yRes, zRes);
	caculateScaleFactor << <block, thread >> > (d_data, d_size, xRes, yRes, zRes);
	caculateAllForce << <block, thread >> > (d_data, d_size, xRes, yRes, zRes, iter);
	caculateCollision << <block, thread >> > (d_data, d_size, xRes, yRes, zRes, d_wall, walls.size());
	//计算新的位置
	updatePosition << <block, thread >> > (d_data, d_size, xRes, yRes, zRes, iter);
	//重新划分网格
	//  准备new_data, new_lock, new_size的内存空间
	CHECK(cudaMemcpy(h_data, d_data, sizeof(particle*) * xRes*yRes*zRes, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(h_size, d_size, sizeof(int) * xRes*yRes*zRes, cudaMemcpyDeviceToHost));
	/*new_size = (int*)malloc(sizeof(int) * xRes * yRes * zRes);
	memset(new_size, 0, sizeof(int)  * xRes * yRes * zRes);
	particle** hnew_data = (particle**)malloc(xRes * yRes * zRes * sizeof(particle*));
	for (int x = 0; x < xRes; x++) {
		for (int y = 0; y < yRes; y++) {
			for (int z = 0; z < zRes; z++) {
				//分配GPU空间存储粒子
				particle* h_temp = (particle*)malloc(MAXINGRID * sizeof(particle));;
				particle* d2h_temp = (particle*)malloc(MAXINGRID * sizeof(particle));
				//CHECK(cudaMalloc((void**)&d_temp, h_size[grid_index(x, y, z, xRes, yRes, zRes)] * sizeof(particle)));
				memset(h_temp, 0, MAXINGRID * sizeof(particle));
				memset(d2h_temp, 0, MAXINGRID * sizeof(particle));
				CHECK(cudaMemcpy(d2h_temp, h_data[grid_index(x, y, z, xRes, yRes, zRes)], MAXINGRID * sizeof(particle), cudaMemcpyDeviceToHost));
				hnew_data[grid_index(x, y, z, xRes, yRes, zRes)] = h_temp;
				h_data[grid_index(x, y, z, xRes, yRes, zRes)] = d2h_temp;
			}//grid.x
		}//grid.y
	}//grid.z*/
	//   On CPU
	/*updataGridOnHost(h_data, h_size, xRes, yRes, zRes, hnew_data, new_size);
	for (int i = 0; i < xRes * yRes * zRes; ++i) {
		free(h_data[i]);
	}
	free(h_data);
	h_data = hnew_data;
	free(h_size);
	h_size = new_size;*/
	//struct to class copy
	
	for (int x = 0; x < (*field).xRes(); x++) {
		for (int y = 0; y < (*field).yRes(); y++) {
			for (int z = 0; z < (*field).zRes(); z++) {
				vector<PARTICLE>& par = (*field)(x, y, z);
				vector<PARTICLE>::iterator itc = par.begin();
				h_size[grid_index(x, y, z, xRes, yRes, zRes)] = par.size();
				//分配CPU空间存储粒子
				particle* h_temp = (particle*)malloc(h_size[grid_index(x, y, z, xRes, yRes, zRes)] * sizeof(particle));;
				CHECK(cudaMemcpy(h_temp, h_data[grid_index(x, y, z, xRes, yRes, zRes)], sizeof(particle) * h_size[grid_index(x, y, z, xRes, yRes, zRes)], cudaMemcpyDeviceToHost));
				CHECK(cudaFree(h_data[grid_index(x, y, z, xRes, yRes, zRes)]));
				h_data[grid_index(x, y, z, xRes, yRes, zRes)] = h_temp;
				for (int p = 0; p < h_size[grid_index(x, y, z, xRes, yRes, zRes)]; ++p) {
					if (abs(h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].position.x) > DX || abs(h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].position.z) > DZ) {
						itc = par.erase(itc);
						continue;
					}
					itc->acceleration().x = h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].acceleration.x;
					itc->acceleration().y = h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].acceleration.y;
					itc->acceleration().z = h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].acceleration.z;
					itc->position().x = h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].position.x;
					itc->position().y = h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].position.y;
					itc->position().z = h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].position.z;
					itc->velocity().x = h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].velocity.x;
					itc->velocity().y = h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].velocity.y;
					itc->velocity().z = h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].velocity.z;
					itc->flag() = h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].flag;
					itc->id() = h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].id;
					itc->exSPHflag() = h_data[grid_index(x, y, z, xRes, yRes, zRes)][p].exSPHflag;
					++itc;
				}
			}//grid.x
		}//grid.y
	}//grid.z
	/*for (int x = 0; x < xRes; x++) {
		for (int y = 0; y < yRes; y++) {
			for (int z = 0; z < zRes; z++) {
				particle* d_temp;
				int idx = grid_index(x, y, z, xRes, yRes, zRes);
				CHECK(cudaMalloc((void**)&d_temp, MAXINGRID * sizeof(particle)));
				CHECK(cudaMemset(d_temp, 0, MAXINGRID * sizeof(particle)));
				CHECK(cudaMemcpy(d_temp, h_data[idx], sizeof(particle) * MAXINGRID, cudaMemcpyHostToDevice));
				free(h_data[idx]);
				h_data[idx] = d_temp;
			}//grid.x
		}//grid.y
	}//grid.z
	CHECK(cudaMemcpy(d_size, h_size, sizeof(int) * xRes*yRes*zRes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_data, h_data, sizeof(particle*) * xRes*yRes*zRes, cudaMemcpyHostToDevice));*/
	for (int i = 0; i < xRes*yRes*zRes; ++i)
		free(h_data[i]);
	return;
}

void remalloc(int _xRes, int _yRes, int _zRes) {
	if (_xRes != xRes || _yRes != yRes || _zRes != zRes) {
		CHECK(cudaMalloc((void**)&d_data, sizeof(particle*) * xRes * yRes * zRes));
	}
}