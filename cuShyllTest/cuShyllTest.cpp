#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

#define MAX_DEPTH 100

#pragma region RANDOM
__host__ __device__ float randLCG(uint32_t* seed)
{
	*seed = 1664525 * *seed + 1013904223;
	return *seed / float(UINT32_MAX);
}

__host__ __device__ float randXORShift(uint32_t* seed)
{
	*seed ^= (*seed << 13);
	*seed ^= (*seed >> 17);
	*seed ^= (*seed << 5);
	return *seed / float(UINT32_MAX);
}

__host__ __device__ void wangHash(uint32_t* seed)
{
	*seed = (*seed ^ 61) ^ (*seed >> 16);
	*seed *= 9;
	*seed = *seed ^ (*seed >> 4);
	*seed *= 0x27d4eb2d;
	*seed = *seed ^ (*seed >> 15);
}
#pragma endregion

#pragma region VEC3
struct Vec3
{
	float x;
	float y;
	float z;

	__host__ __device__ Vec3()
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}

	__host__ __device__ Vec3(float val)
	{
		x = val;
		y = val;
		z = val;
	}

	__host__ __device__ Vec3(float _x, float _y, float _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

	__host__ __device__ inline float Length() const { return sqrt(LengthSquared()); }
	__host__ __device__ inline float LengthSquared() const { return x * x + y * y + z * z; }
	__host__ __device__ inline Vec3 Normalized() const;
	__host__ __device__ inline Vec3 RotateX(float cosTheta, float sinTheta) const;
	__host__ __device__ inline Vec3 RotateY(float cosTheta, float sinTheta) const;
	__host__ __device__ inline Vec3 RotateZ(float cosTheta, float sinTheta) const;

	__host__ __device__ inline static float Dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
	__host__ __device__ inline static Vec3 Cross(Vec3 a, Vec3 b) { return Vec3(a.y * b.z - a.z * b.y, -(a.x * b.z - a.z * b.x), a.x * b.y - a.y * b.x); }
	__host__ __device__ inline static Vec3 RandomUnitVector(unsigned int* seed);
};

__host__ __device__ inline Vec3 operator -(Vec3 a) { return Vec3(-a.x, -a.y, -a.z); }
__host__ __device__ inline Vec3 operator +(Vec3 a, Vec3 b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ inline Vec3 operator -(Vec3 a, Vec3 b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ inline Vec3 operator *(Vec3 a, Vec3 b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
__host__ __device__ inline Vec3 operator /(Vec3 a, Vec3 b) { return Vec3(a.x / b.x, a.y / b.y, a.z / b.z); }
__host__ __device__ inline Vec3 operator *(Vec3 vec, float scalar) { return Vec3(vec.x * scalar, vec.y * scalar, vec.z * scalar); }
__host__ __device__ inline Vec3 operator /(Vec3 vec, float scalar) { return Vec3(vec.x / scalar, vec.y / scalar, vec.z / scalar); }
__host__ __device__ inline Vec3 operator *(float scalar, Vec3 vec) { return Vec3(vec.x * scalar, vec.y * scalar, vec.z * scalar); }
__host__ __device__ inline Vec3 operator /(float scalar, Vec3 vec) { return Vec3(vec.x / scalar, vec.y / scalar, vec.z / scalar); }

__host__ __device__ inline Vec3 Vec3::RotateX(float cosTheta, float sinTheta) const
{
	float ny = cosTheta * y - sinTheta * z;
	float nz = sinTheta * y + cosTheta * z;
	return Vec3(x, ny, nz);
}

__host__ __device__ inline Vec3 Vec3::RotateY(float cosTheta, float sinTheta) const
{
	float nx = cosTheta * x - sinTheta * z;
	float nz = sinTheta * x + cosTheta * z;
	return Vec3(nx, y, nz);
}

__host__ __device__ inline Vec3 Vec3::RotateZ(float cosTheta, float sinTheta) const
{
	float nx = cosTheta * x - sinTheta * y;
	float ny = sinTheta * x + cosTheta * y;
	return Vec3(nx, ny, z);
}

__host__ __device__ inline Vec3 Vec3::Normalized() const { return Vec3(x, y, z) / Length(); }
__host__ __device__ inline Vec3 Vec3::RandomUnitVector(uint32_t* seed)
{
	return (2.0f * Vec3(randLCG(seed), randLCG(seed), randLCG(seed)) - Vec3(1.0f)).Normalized();
}
#pragma endregion

#pragma region RAY3
struct Ray3
{
	__host__ __device__ Ray3(Vec3 o, Vec3 d)
	{
		origin = o;
		direction = d.Normalized();
	}

	__host__ __device__ Vec3 PointAt(float t) const { return origin + t * direction; }

	__host__ __device__ Vec3 Origin() const { return origin; }
	__host__ __device__ Vec3 Direction() const { return direction; }

	__host__ __device__ Ray3 RotateX(float cosTheta, float sinTheta) const
	{
		return Ray3(origin.RotateX(cosTheta, sinTheta), direction.RotateX(cosTheta, sinTheta));
	}
	__host__ __device__ Ray3 RotateY(float cosTheta, float sinTheta) const
	{
		return Ray3(origin.RotateY(cosTheta, sinTheta), direction.RotateY(cosTheta, sinTheta));
	}
	__host__ __device__ Ray3 RotateZ(float cosTheta, float sinTheta) const
	{
		return Ray3(origin.RotateZ(cosTheta, sinTheta), direction.RotateZ(cosTheta, sinTheta));
	}

private:
	Vec3 origin, direction;
};
#pragma endregion

#pragma region CAMERA
struct Camera
{
	__host__ __device__ Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vup, float vfov, float aspect)
	{
		Vec3 u, v, w;
		float hh = tan(vfov / 2.0f);
		float hw = aspect * hh;
		origin = lookFrom;
		w = (lookFrom - lookAt).Normalized();
		u = Vec3::Cross(vup, w).Normalized();
		v = Vec3::Cross(w, u);
		llc = origin - hw * u - hh * v - w;
		horizontal = 2.0f * hw * u;
		vertical = 2.0f * hh * v;
	}

	__host__ __device__ inline Ray3 GetRay(float u, float v) const { return Ray3(origin, llc + u * horizontal + v * vertical - origin); }

private:
	Vec3 origin, llc, horizontal, vertical;
};
#pragma endregion

#pragma region SAVEIMAGE
#define CLAMP(x, min, max) ((x < min) ? min : ((x > max) ? max : x))
void saveImage(int width, int height, Vec3* colors, const char* fname)
{
	std::ofstream file;
	file.open(fname, std::ios::binary | std::ios::trunc);
	file << "P6" << std::endl;
	file << std::to_string(width) << ' ' << std::to_string(height) << std::endl;
	file << "255" << std::endl;

	for (int j = height - 1; j >= 0; j--)
	{
		for (int i = 0; i < width; i++)
		{
			float r = sqrt(colors[i + j * width].x);
			float g = sqrt(colors[i + j * width].y);
			float b = sqrt(colors[i + j * width].z);
			r = CLAMP(r, 0.0f, 1.0f);
			g = CLAMP(g, 0.0f, 1.0f);
			b = CLAMP(b, 0.0f, 1.0f);
			int8_t ir = int8_t(255.0f * r);
			int8_t ig = int8_t(255.0f * g);
			int8_t ib = int8_t(255.0f * b);
			file << ir << ig << ib;
		}
	}

	file.close();
}
#undef CLAMP
#pragma endregion

#include "cuShyllGenerated.txt"

__device__ bool listHit(Hittable* hittables, int numHittables, Ray3& ray, float tMin, float tMax, float& t, Vec3& point, Vec3& normal, Material*& mat)
{
	float tempt;
	Vec3 tempp;
	Vec3 tempn;
	Material* tempm;
	bool hitAnything = false;
	for (int i = 0; i < numHittables; i++)
	{
		if (hittables[i].Hit(ray, tMin, tMax, tempt, tempp, tempn, tempm))
		{
			hitAnything = true;
			tMax = tempt;
			t = tempt;
			point = tempp;
			normal = tempn;
			mat = tempm;
		}
	}
	return hitAnything;
}

__device__ Vec3 color(uint32_t* seed, Hittable* hittables, int numHittables, Ray3& ray)
{
	float t;
	Vec3 point;
	Vec3 normal;
	Material* mat;
	int depth = 0;
	Vec3 sum(0.0f);
	Vec3 multiplier(1.0f);
	Vec3 attenuation;

	while (depth++ < MAX_DEPTH)
	{
		if (listHit(hittables, numHittables, ray, 0.001f, FLT_MAX, t, point, normal, mat))
		{
			// todo: probably find out why texture pointer was changing when passed to function??
			sum = sum + mat->Emit(seed, t, point, normal, ray) * multiplier;
			if (mat->Scatter(seed, t, point, normal, ray, attenuation))
			{
				multiplier = multiplier * attenuation;
			}
			else
			{
				break;
			}
		}
	}
	return sum;
}

__global__ void render(Hittable* hittables, int numHittables, Camera cam, Vec3* cols, int width, int height, int samples, int sx, int sy, int ex, int ey)
{
	for (int i = sx + (blockIdx.x * blockDim.x) + threadIdx.x; i < ex; i += blockDim.x * gridDim.x)
	{
		if (i >= width) continue;
		for (int j = sy + (blockIdx.y * blockDim.y) + threadIdx.y; j < ey; j += blockDim.y * gridDim.y)
		{
			if (j >= height) continue;
			uint32_t seed = (i ^ j) + ((threadIdx.x + i + 1) * (threadIdx.y + j + 1));
			wangHash(&seed);

			Vec3 overallCol;

			for (int s = 0; s < samples; s++)
			{
				float u = (i + randLCG(&seed)) / float(width), v = (j + randLCG(&seed)) / float(height);
				Ray3 ray = cam.GetRay(u, v);
				wangHash(&seed);
				overallCol = overallCol + color(&seed, hittables, numHittables, ray);
			}

			cols[i + j * width] = overallCol / float(samples);
		}
	}
}

auto startTime = std::chrono::high_resolution_clock::now();

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d %lld\n", cudaGetErrorString(code), file, line, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count());
		if (abort)
		{
			cudaDeviceReset();
			exit(code);
		}
	}
}

int main()
{
	const int width = 600, height = 600, samples = 1000;

	Vec3* cols;
	gpuErrchk(cudaMallocManaged(&cols, width * height * sizeof(Vec3)));

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(8, 8);

	Camera cam = Camera(Vec3(0.0f, 0.7f, -2.5f), Vec3(0.0f, 0.1f, 2.5f), Vec3(0.0f, 1.0f, 0.0f), 1.2f, width / float(height));

	Texture* tlist;
	gpuErrchk(cudaMallocManaged(&tlist, 4 * sizeof(Texture)));
	tlist[0] = ConstantColor(Vec3(1.0f, 0.2f, 0.0f));
	tlist[1] = ConstantColor(Vec3(0.2f, 0.8f, 0.4f));
	tlist[2] = ConstantColor(Vec3(0.8f, 0.6f, 0.65f));
	tlist[3] = ConstantColor(Vec3(2.5f));

	Material* mlist;
	gpuErrchk(cudaMallocManaged(&mlist, 4 * sizeof(Material)));
	mlist[0] = Lambertian(&tlist[0]);
	mlist[1] = Lambertian(&tlist[1]);
	mlist[2] = Dieletric(1.8f, &tlist[2]);
	mlist[3] = DiffuseLight(&tlist[3]);

	Hittable* hlist;
	gpuErrchk(cudaMallocManaged(&hlist, 4 * sizeof(Hittable)));
	hlist[0] = Sphere(Vec3(0.55f, 0.5f, 0.0f), 0.5f, &mlist[0]);
	hlist[1] = Sphere(Vec3(0.0f, -100.0f, 0.0f), 100.0f, &mlist[1]);
	hlist[2] = Sphere(Vec3(-0.55f, 0.5f, 0.0f), 0.5f, &mlist[2]);
	hlist[3] = Sphere(Vec3(0.0f, 2.5f, -0.2f), 1.5f, &mlist[3]);

	for (int i = 0; i < width; i += threadsPerBlock.x * numBlocks.x / static_cast<int>(log10(samples)))
	{
		for (int j = 0; j < height; j += threadsPerBlock.y * numBlocks.y / static_cast<int>(log10(samples)))
		{
			render<<<numBlocks, threadsPerBlock>>>(hlist, 4, cam, cols, width, height, samples, i, j, i + threadsPerBlock.x * numBlocks.x / 2, j + threadsPerBlock.y * numBlocks.y / 2);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	saveImage(width, height, cols, "test2.ppm");
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count() << "\n";

	cudaDeviceReset();

	return 0;
}