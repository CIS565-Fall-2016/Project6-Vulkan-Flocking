#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

// LOOK: corresponds with outColor from particle.vert
layout (location = 0) in vec2 inColor;

layout (location = 0) out vec4 outFragColor;

void main ()
{
	outFragColor.rgb = vec3(inColor.x, abs(inColor.y), -inColor.x) * 10.0;
}
