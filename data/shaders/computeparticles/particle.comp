#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

struct Particle
{
	vec2 pos;
	vec2 vel;
};

// LOOK: These bindings correspond to the DescriptorSetLayouts and
// the DescriptorSets from prepareCompute()!

// Binding 0 : Particle storage buffer (read)
layout(std140, binding = 0) buffer ParticlesA
{
   Particle particlesA[ ];
};

// Binding 1 : Particle storage buffer (write)
layout(std140, binding = 1) buffer ParticlesB
{
   Particle particlesB[ ];
};

layout (local_size_x = 16, local_size_y = 16) in;

// LOOK: rule weights and distances, as well as particle count, based off uniforms.
// The deltaT here has to be updated every frame to account for changes in
// frame rate.
layout (binding = 2) uniform UBO
{
	float deltaT;
	float rule1Distance;
	float rule2Distance;
	float rule3Distance;
	float rule1Scale;
	float rule2Scale;
	float rule3Scale;
	int particleCount;
} ubo;

void main()
{
		// LOOK: This is very similar to a CUDA kernel.
		// Right now, the compute shader only advects the particles with their
		// velocity and handles wrap-around.
		// TODO: implement flocking behavior.

    // Current SSBO index
    uint index = gl_GlobalInvocationID.x;
	// Don't try to write beyond particle count
    if (index >= ubo.particleCount)
		return;

    // Read position and velocity
		vec2 vPos = particlesA[index].pos.xy;
    vec2 vVel = particlesA[index].vel.xy;

		// clamp velocity for a more pleasing simulation.
		vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);

		// kinematic update
		vPos += vVel * ubo.deltaT;

    // Wrap around boundary
		if (vPos.x < -1.0) vPos.x = 1.0;
		if (vPos.x > 1.0) vPos.x = -1.0;
		if (vPos.y < -1.0) vPos.y = 1.0;
		if (vPos.y > 1.0) vPos.y = -1.0;

    particlesB[index].pos.xy = vPos;

    // Write back
    particlesB[index].vel.xy = vVel;
}
