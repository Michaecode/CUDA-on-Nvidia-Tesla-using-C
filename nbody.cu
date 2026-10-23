#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Body { float x, y, z, mass; float vx, vy, vz, pad; };
struct Acceleration { float x, y, z; };

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// One CUDA thread calculates the acceleration of one body.
__global__ void calculate_accelerations(const Body* bodies, Acceleration* accelerations, int count, float G, float softening_squared) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Body self = bodies[i];
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int j = 0; j < count; ++j) {
        if (j == i) continue;
        const Body other = bodies[j];
        const float dx = other.x - self.x, dy = other.y - self.y, dz = other.z - self.z;
        const float r2 = dx * dx + dy * dy + dz * dz + softening_squared;
        const float inv_r = rsqrtf(r2);
        const float scale = G * other.mass * inv_r * inv_r * inv_r;
        ax += dx * scale; ay += dy * scale; az += dz * scale;
    }
    accelerations[i] = {ax, ay, az};
}

// Separate integration kernel ensures all forces use positions from one timestep.
__global__ void integrate(Body* bodies, const Acceleration* accelerations, int count, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    Body b = bodies[i];
    const Acceleration a = accelerations[i];
    b.vx += a.x * dt; b.vy += a.y * dt; b.vz += a.z * dt;
    b.x += b.vx * dt; b.y += b.vy * dt; b.z += b.vz * dt;
    bodies[i] = b;
}

static void write_csv(const std::string& path, const std::vector<Body>& bodies) {
    std::ofstream out(path);
    if (!out) { std::cerr << "Error: cannot open output file: " << path << '\n'; std::exit(EXIT_FAILURE); }
    out << "x,y,z,vx,vy,vz,mass\n" << std::setprecision(9);
    for (const Body& b : bodies) out << b.x << ',' << b.y << ',' << b.z << ',' << b.vx << ',' << b.vy << ',' << b.vz << ',' << b.mass << '\n';
}

int main(int argc, char** argv) {
    int count = 4096, steps = 100;
    float dt = 0.001f;
    const float G = 1.0f, softening = 0.01f;
    std::string output_path;
    try {
        if (argc > 1) count = std::stoi(argv[1]);
        if (argc > 2) steps = std::stoi(argv[2]);
        if (argc > 3) dt = std::stof(argv[3]);
        if (argc > 4) output_path = argv[4];
    } catch (const std::exception&) {
        std::cerr << "Usage: " << argv[0] << " [bodies=4096] [steps=100] [dt=0.001] [output.csv]\n";
        return EXIT_FAILURE;
    }
    if (count < 1 || steps < 0 || !std::isfinite(dt) || dt <= 0.0f) {
        std::cerr << "Error: bodies must be >= 1, steps >= 0, and dt finite and > 0.\n";
        return EXIT_FAILURE;
    }
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << "\nBodies: " << count << " | steps: " << steps << " | dt: " << dt << '\n';

    std::vector<Body> host(static_cast<size_t>(count));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos(-1.0f, 1.0f), vel(-0.05f, 0.05f), mass(0.5f, 1.5f);
    for (Body& b : host) { b.x = pos(rng); b.y = pos(rng); b.z = pos(rng); b.mass = mass(rng); b.vx = vel(rng); b.vy = vel(rng); b.vz = vel(rng); b.pad = 0.0f; }

    Body* d_bodies = nullptr;
    Acceleration* d_acc = nullptr;
    const size_t body_bytes = host.size() * sizeof(Body), acc_bytes = static_cast<size_t>(count) * sizeof(Acceleration);
    CUDA_CHECK(cudaMalloc(&d_bodies, body_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc, acc_bytes));
    CUDA_CHECK(cudaMemcpy(d_bodies, host.data(), body_bytes, cudaMemcpyHostToDevice));
    constexpr int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int step = 0; step < steps; ++step) {
        calculate_accelerations<<<blocks, threads>>>(d_bodies, d_acc, count, G, softening * softening);
        CUDA_CHECK(cudaGetLastError());
        integrate<<<blocks, threads>>>(d_bodies, d_acc, count, dt);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    CUDA_CHECK(cudaMemcpy(host.data(), d_bodies, body_bytes, cudaMemcpyDeviceToHost));
    std::cout << std::fixed << std::setprecision(3) << "Simulation time: " << elapsed << " ms";
    if (steps > 0) std::cout << " (" << elapsed / steps << " ms/step)";
    std::cout << '\n';
    if (!output_path.empty()) { write_csv(output_path, host); std::cout << "Final state written to: " << output_path << '\n'; }
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_acc)); CUDA_CHECK(cudaFree(d_bodies)); CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
