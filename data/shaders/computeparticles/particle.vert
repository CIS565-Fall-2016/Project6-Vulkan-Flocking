#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

// LOOK: These correspond with the attributeDescriptions in
// prepareStorageBuffers()!
layout (location = 0) in vec2 inPos;
layout (location = 1) in vec2 inVel;

// LOOK: out to rasterization, then to the `in` layouts in particle.frag
layout (location = 0) out vec2 outColor;

// emit a point to rasterization from each thread running particle.vert
out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

void main ()
{
  gl_PointSize = 2.0;
  outColor = inVel;
  gl_Position = vec4(inPos.xy, 1.0, 1.0);
}
