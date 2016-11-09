/*
* Vulkan Example - Boids based compute shader particle system
*
* Based on Vulkan Examples by Sascha Willems.
* Modified to Boids by Kangning Gary Li
* for CIS 565: GPU Programming, with Patrick Cozzi.
* University of Pennsylvania, Fall 2016.
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <random>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION true // LOOK: toggle Vulkan validation layers. These make debugging much easier!
#define PARTICLE_COUNT 4 * 1024 // LOOK: change particle count here

// LOOK: constants for the boids algorithm. These will be passed to the GPU compute part of the assignment
// using a Uniform Buffer. These parameters should yield a stable and pleasing simulation for an
// implementation based off the code here: http://studio.sketchpad.cc/sp/pad/view/ro.9cbgCRcgbPOI6/rev.23
#define RULE1DISTANCE 0.1f // cohesion
#define RULE2DISTANCE 0.05f // separation
#define RULE3DISTANCE 0.05f // alignment
#define RULE1SCALE 0.02f
#define RULE2SCALE 0.05f
#define RULE3SCALE 0.01f

class VulkanExample : public VulkanExampleBase
{
public:
	float timer = 0.0f;
	float animStart = 20.0f;
	bool animate = true;

	// LOOK: this struct contains descriptions of how the vertex buffer should be interpreted by
	// a strictly graphics pipeline. 
	struct {
		// inputState encapsulates bindingDescriptions and  attributeDescriptions
		VkPipelineVertexInputStateCreateInfo inputState;
		// bindingDescriptions describe how a buffer should be broken down into structs
		std::vector<VkVertexInputBindingDescription> bindingDescriptions;
		// attributeDescriptions describe how structs from the buffer should be
		// reinterpreted as attributes in the shader program.
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	// LOOK: more descriptions of how to attach things to the graphics pipeline.
	// These are important in Sascha Willems's original compute particles example
	// because his pipeline took texture inputs. Don't worry too much about these for now.
	struct {
		VkDescriptorSetLayout descriptorSetLayout;	// Particle system rendering shader binding layout
		VkDescriptorSet descriptorSet;				// Particle system rendering shader bindings
		VkPipelineLayout pipelineLayout;			// Layout of the graphics pipeline
		VkPipeline pipeline;						// Particle rendering pipeline
	} graphics;

	// LOOK: Resources for the compute part of the example
	struct {
		vk::Buffer storageBufferA;					// (Shader) storage buffer object containing the particles
		vk::Buffer storageBufferB;					// (Shader) storage buffer object containing the particles

		vk::Buffer uniformBuffer;					// Uniform buffer object containing particle system parameters
		VkQueue queue;								// Separate queue for compute commands (queue family may differ from the one used for graphics)
		VkCommandPool commandPool;					// Use a separate command pool (queue family may differ from the one used for graphics)
		VkCommandBuffer commandBuffer;				// Command buffer storing the dispatch commands and barriers
		VkFence fence;								// Synchronization fence to avoid rewriting compute CB if still in use

		VkDescriptorSetLayout descriptorSetLayout;	// Compute shader binding layout - how to interface with the pipeline
		VkDescriptorSet descriptorSets[2];			// Compute shader bindings - encapsulate buffers for interfacing with the pipeline
													// in acoordance with the descriptorSetLayout
		VkPipelineLayout pipelineLayout;			// Layout of the compute pipeline
		VkPipeline pipeline;						// Compute pipeline for updating particle positions

		struct computeUBO {							// Compute shader uniform block object
			float deltaT;							// Frame delta time
			float rule1Distance = RULE1DISTANCE;	// Boids rule parameters
			float rule2Distance = RULE2DISTANCE;
			float rule3Distance = RULE3DISTANCE;
			float rule1Scale = RULE1SCALE;
			float rule2Scale = RULE2SCALE;
			float rule3Scale = RULE3SCALE;
			int32_t particleCount = PARTICLE_COUNT;
		} ubo;
	} compute;

	// LOOK: This is what our particles will look like and how they will
	// be packed in memory.
	struct Particle {
		glm::vec2 pos;								// Particle position
		glm::vec2 vel;								// Particle velocity
	};

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		enableTextOverlay = true;
		title = "Vulkan Example - Compute shader particle system";
	}

	~VulkanExample()
	{
		// Graphics
		vkDestroyPipeline(device, graphics.pipeline, nullptr);
		vkDestroyPipelineLayout(device, graphics.pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, graphics.descriptorSetLayout, nullptr);

		// Compute
		compute.storageBufferA.destroy();
		compute.storageBufferB.destroy();

		compute.uniformBuffer.destroy();
		vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
		vkDestroyPipeline(device, compute.pipeline, nullptr);
		vkDestroyFence(device, compute.fence, nullptr);
		vkDestroyCommandPool(device, compute.commandPool, nullptr);
	}

	//////// Setup Functions ////////

	void prepare()
	{
		VulkanExampleBase::prepare();
		prepareStorageBuffers();
		prepareUniformBuffers(); // uniform buffers for compute
		setupDescriptorSetLayout(); // for the graphics pipeline
		preparePipelines();
		setupDescriptorPool();
		prepareCompute();
		prepared = true;
	}

	// Setup and fill the compute shader storage buffers containing the particles
	void prepareStorageBuffers()
	{
		// LOOK: generate the buffers CPU-side using a RNG. We'll just do all work in screen space.

		std::mt19937 rGenerator;
		std::uniform_real_distribution<float> rDistribution(-1.0f, 1.0f);

		// Initial particle positions
		std::vector<Particle> particleBuffer(PARTICLE_COUNT);
		for (auto& particle : particleBuffer)
		{
			particle.pos = glm::vec2(rDistribution(rGenerator), rDistribution(rGenerator));
			// TODO: add randomized velocities with a slight scale here, something like 0.1f.
		}

		VkDeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);

		// LOOK: Transfer the data to the GPU using a staging buffer.
		// Read through the comments to the next LOOK spot

		// We will transfer first to memory visible to both CPU and GPU, then to memory visible to the GPU.
		// This involves creating an intermediate "staging buffer."
		// All these buffers are vk::Buffers, but the GPU decides what kind of memory each is backed by
		// when you describe what you are going to do with it.

		vk::Buffer stagingBuffer;

		// Sascha Willems abstracted away a lot of the buffer creation nastiness for us!
		// Buffer availability is dependent on queue, wich depends on logical device, which depends on physical device...
		// We will assume Vulkan sees your GPU as a single device, to make things simpler.

		// transfer buffer -> first use slow, CPU and GPU accessible memory to get our positions onto the GPU at all
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT, // make a buffer to be used for transfers
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, // out of host visible memory
			&stagingBuffer,
			storageBufferSize,
			particleBuffer.data()); // transfer data from the host

		// These SSBOs will be used as storage buffers for the compute pipeline and as a vertex buffer in the graphics pipeline
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&compute.storageBufferA,
			storageBufferSize); // no data from host - not visible to host

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&compute.storageBufferB,
			storageBufferSize);

		// Copy from staging buffer (slow GPU/CPU memory) to actual buffer (faster GPU local memory)
		// To copy from the staging buffer to one of our SSBOs, we will need to create a command buffer on
		// this application's GPU command queue.
		// Getting the GPU command queue and creating the copy instruction is handled for us by the base code.
		VkCommandBuffer copyCmd = VulkanExampleBase::createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = storageBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, compute.storageBufferA.buffer, 1, &copyRegion);
		// Run all commands in the queue and wait for the queue to become idle before returning.
		// See implementation below for details - it basically "while" loops until the queue is empty.
		// Have multiple transfers? Can run them simultaneously by using
		// fences instead of waiting for each to finish before starting the next one.
		VulkanExampleBase::flushCommandBuffer(copyCmd, queue, true);

		// destroy the staging buffer, don't need it anymore.
		stagingBuffer.destroy();

		//// LOOK: set up descriptors for getting graphics pipeline to use our new SSBOs ////
		// read comments through to the end of the function.

		// Binding description - How many bits go to each shader "thread?" What's the stride?
		// We will only have a single descriptor, since our graphics pipeline is very simple.
		// The vertexInputBindingDescription basically ensures that our SSBO is interpreted
		// by the graphics pipeline as:
		vertices.bindingDescriptions.resize(1);
		vertices.bindingDescriptions[0] =
			vkTools::initializers::vertexInputBindingDescription(
			VERTEX_BUFFER_BIND_ID,        // * vertex data
			sizeof(Particle),	          // * with a full Particle struct handed to each "thread"
			VK_VERTEX_INPUT_RATE_VERTEX);

		// Attribute descriptions - Which bits in the per-shader thread struct go where in the shader?
		// Describes memory layout and shader positions
		vertices.attributeDescriptions.resize(2);
		// Location 0 : Position
		vertices.attributeDescriptions[0] =
			vkTools::initializers::vertexInputAttributeDescription(
			VERTEX_BUFFER_BIND_ID,
			0, // corresponds to `layout (location = 0) in` in particle.vert 
			VK_FORMAT_R32G32_SFLOAT, // what kind of data? vec2
			offsetof(Particle, pos)); // offset into each Particle struct
		// Location 1 : Velocity
		vertices.attributeDescriptions[1] =
			vkTools::initializers::vertexInputAttributeDescription(
			VERTEX_BUFFER_BIND_ID,
			1,
			VK_FORMAT_R32G32_SFLOAT,
			offsetof(Particle, pos)); // TODO: change this so that we can color the particles based on velocity.

		// vertices.inputState encapsulates everything we need for these particular buffers to
		// interface with the graphics pipeline.
		vertices.inputState = vkTools::initializers::pipelineVertexInputStateCreateInfo();
		vertices.inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertices.bindingDescriptions.size());
		vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
		vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.attributeDescriptions.size());
		vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
	}

	// Prepare and initialize uniform buffer containing shader uniforms for compute
	void prepareUniformBuffers()
	{
		// Compute shader uniform buffer block
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&compute.uniformBuffer,
			sizeof(compute.ubo));

		// Map for host access
		VK_CHECK_RESULT(compute.uniformBuffer.map());

		updateUniformBuffers();
	}

	void setupDescriptorSetLayout() // for graphics pipeline. Not very important for this assignment.
	{
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;

		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vkTools::initializers::descriptorSetLayoutCreateInfo(
			setLayoutBindings.data(),
			static_cast<uint32_t>(setLayoutBindings.size()));

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &graphics.descriptorSetLayout));

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
			vkTools::initializers::pipelineLayoutCreateInfo(
			&graphics.descriptorSetLayout,
			1);

		// layout bound to pipeline here. descriptor set can be set up later to match.
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &graphics.pipelineLayout));
	}

	// LOOK: Create the graphics pipeline. Skim to the next LOOK
	void preparePipelines()
	{
		// dscribe a bunch of pipeline state
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vkTools::initializers::pipelineInputAssemblyStateCreateInfo(
			VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
			0,
			VK_FALSE);

		VkPipelineRasterizationStateCreateInfo rasterizationState =
			vkTools::initializers::pipelineRasterizationStateCreateInfo(
			VK_POLYGON_MODE_FILL,
			VK_CULL_MODE_NONE,
			VK_FRONT_FACE_COUNTER_CLOCKWISE,
			0);

		VkPipelineColorBlendAttachmentState blendAttachmentState =
			vkTools::initializers::pipelineColorBlendAttachmentState(
			0xf,
			VK_FALSE);

		VkPipelineColorBlendStateCreateInfo colorBlendState =
			vkTools::initializers::pipelineColorBlendStateCreateInfo(
			1,
			&blendAttachmentState);

		VkPipelineDepthStencilStateCreateInfo depthStencilState =
			vkTools::initializers::pipelineDepthStencilStateCreateInfo(
			VK_FALSE,
			VK_FALSE,
			VK_COMPARE_OP_ALWAYS);

		VkPipelineViewportStateCreateInfo viewportState =
			vkTools::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

		VkPipelineMultisampleStateCreateInfo multisampleState =
			vkTools::initializers::pipelineMultisampleStateCreateInfo(
			VK_SAMPLE_COUNT_1_BIT,
			0);

		std::vector<VkDynamicState> dynamicStateEnables = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState =
			vkTools::initializers::pipelineDynamicStateCreateInfo(
			dynamicStateEnables.data(),
			static_cast<uint32_t>(dynamicStateEnables.size()),
			0);

		// Rendering pipeline
		// LOOK: Load shaders. These are the spirv shaders, not the glsl!
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/computeparticles/particle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/computeparticles/particle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		VkGraphicsPipelineCreateInfo pipelineCreateInfo =
			vkTools::initializers::pipelineCreateInfo(
			graphics.pipelineLayout,
			renderPass,
			0);

		// LOOK: set the pipeline up to interface with our buffers using the
		// inputState from prepareStorageBuffers()
		pipelineCreateInfo.pVertexInputState = &vertices.inputState; // indicate to pipeline how to use vertex buffer.
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState; // speculation: is this b/c on some GPUs, vertex buffer input is still semi-fixed-function?
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.renderPass = renderPass;

		// Additive blending
		blendAttachmentState.colorWriteMask = 0xF;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;

		// Instruct Vulkan to create the graphics pipeline on the GPU!
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &graphics.pipeline));
	}

	// Descriptor Pools - from these, can allocate descriptor sets, which describe different ways of using data with pipelines
	// Pipelines must have compatible descriptor layout
	void setupDescriptorPool()
	{
		std::vector<VkDescriptorPoolSize> poolSizes =
		{
			vkTools::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5), // can allocate uniform descriptors from here
			vkTools::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5) // can allocate buffer descriptors from here
		};

		VkDescriptorPoolCreateInfo descriptorPoolInfo =
			vkTools::initializers::descriptorPoolCreateInfo( // create separate pools for different types of descriptors?
			static_cast<uint32_t>(poolSizes.size()),
			poolSizes.data(),
			10); // excessively sized for now...

		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	// LOOK: prepare how to interface with the compute pipeline and the pipeline itself.
	// Read through all comments in this function.
	void prepareCompute()
	{
		//////// Create a compute capable device queue ////////
		// The VulkanDevice::createLogicalDevice functions finds a compute capable queue and prefers queue families that only support compute
		// Depending on the implementation this may result in different queue family indices for graphics and computes,
		// requiring proper synchronization (see the memory barriers in buildComputeCommandBuffer, they are fences yay)
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.pNext = NULL;
		queueCreateInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		queueCreateInfo.queueCount = 1;
		vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.compute, 0, &compute.queue);


		//////// Create DescriptorSetLayout ////////

		// LOOK: Describe how we want our data to interface with the future compute pipeline.
		// This is done by creating a DescriptorSetLayout.
		// This defines what kinds of DescriptorSets (like packages of args) the compute pipeline
		// will expect.
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Particle position storage buffer 1
			vkTools::initializers::descriptorSetLayoutBinding(
			VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			VK_SHADER_STAGE_COMPUTE_BIT,
			0), // corresponds to `layout(std140, binding = 0) buffer` in particle.comp
			// Binding 1 : Particle position storage buffer 2
			vkTools::initializers::descriptorSetLayoutBinding(
			VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			VK_SHADER_STAGE_COMPUTE_BIT,
			1), // corresponds to `layout(std140, binding = 1) buffer` in particle.comp
			// Binding 2 : Uniform buffer
			vkTools::initializers::descriptorSetLayoutBinding(
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			VK_SHADER_STAGE_COMPUTE_BIT,
			2)
		};

		// Create the descriptor set layout on the GPU.
		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vkTools::initializers::descriptorSetLayoutCreateInfo(
			setLayoutBindings.data(),
			static_cast<uint32_t>(setLayoutBindings.size()));

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

		//////// Create compute pipeline ////////

		// Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)

		VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vkTools::initializers::pipelineLayoutCreateInfo(
			&compute.descriptorSetLayout, // how the pipeline expects args must be described at pipeline creating time
			1);

		// Create the pipeline layout on the GPU
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout));

		// Create pipeline on the GPU - load shader, attach the pipeline layout we just made.
		VkComputePipelineCreateInfo computePipelineCreateInfo = vkTools::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
		computePipelineCreateInfo.stage = loadShader(getAssetPath() + "shaders/computeparticles/particle.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
		VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline));

		//////// Create command pool and command buffer for compute commands ////////

		// Since commandBuffers live in GPU memory, the GPU likes to get memory for storing
		// them from a preallocated command pool. The GPU can optimize the command pool
		// if we describe what kinds of command buffers we want to use it for.

		// Separate command pool as queue family for compute may be different than graphics
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool));

		// Create a command buffer for compute operations from the command pool.
		// We will write and overwrite this command buffer.
		VkCommandBufferAllocateInfo cmdBufAllocateInfo =
			vkTools::initializers::commandBufferAllocateInfo(
			compute.commandPool,
			VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			1);

		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &compute.commandBuffer));

		// Fence for compute Command Buffer synchronization
		VkFenceCreateInfo fenceCreateInfo = vkTools::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
		VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &compute.fence));


		//////// Create descriptorSetLayouts for using our SSBOs with the pipeline ////////

		// set up descriptor sets for interfacing with descriptor layout on the compute pipeline
		// A descriptor set is like a bunch of args packed together for passing into a function on the GPU.
		// Why do we want two descriptor sets for working with the same layout?
		// HINT: something to do with how the boids algorithm works, what information each boid needs.
		VkDescriptorSetLayout setLayouts[2];
		setLayouts[0] = compute.descriptorSetLayout;
		setLayouts[1] = compute.descriptorSetLayout;

		VkDescriptorSetAllocateInfo allocInfo =
			vkTools::initializers::descriptorSetAllocateInfo(
			descriptorPool,
			setLayouts,
			2);

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, compute.descriptorSets));

		std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets =
		{
			// LOOK
			// WriteDescriptorSet writes each of these descriptors into the specified descriptorSet.
			// THese first few are written into compute.descriptorSet[0].
			// Each of these corresponds to a layout binding in the descriptor set layout,
			// which in turn corresponds with something like `layout(std140, binding = 0)` in `particle.comp`.

			// Binding 0 : Particle position storage buffer
			vkTools::initializers::writeDescriptorSet(
			compute.descriptorSets[0], // LOOK: which descriptor set to write to?
			VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			0, // LOOK: which binding in the descriptor set Layout?
			&compute.storageBufferA.descriptor), // LOOK: which SSBO?

			// Binding 1 : Particle position storage buffer
			vkTools::initializers::writeDescriptorSet(
			compute.descriptorSets[0],
			VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			1,
			&compute.storageBufferB.descriptor),

			// Binding 2 : Uniform buffer
			vkTools::initializers::writeDescriptorSet(
			compute.descriptorSets[0],
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			2,
			&compute.uniformBuffer.descriptor)

			// TODO: write the second descriptorSet, using the top for reference.
			// We want the descriptorSets to be used for flip-flopping:
			// on one frame, we use one descriptorSet with the compute pass,
			// on the next frame, we use the other.
			// What has to be different about how the second descriptorSet is written here?
		};

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);
	}

	//////// Runtime Functions ////////

	void draw()
	{
		buildCommandBuffers();

		buildComputeCommandBuffer();

		// Submit graphics commands
		VulkanExampleBase::prepareFrame();

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::submitFrame();

		// LOOK: wait for fence that was submitted with the compute commandBuffer to complete.
		// Then, reset it for the next round of compute
		vkWaitForFences(device, 1, &compute.fence, VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &compute.fence);

		VkSubmitInfo computeSubmitInfo = vkTools::initializers::submitInfo();
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;

		// LOOK: Submit compute commandBuffer with a fence to avoid rewriting the commandBuffer
		// or doing any other compute before all the instructions in this commandBuffer
		// are done executing.
		VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, compute.fence));

		// TODO: handle flip-flop logic. We want the next iteration to
		// run the compute pipeline with flipped SSBOs, so we have to
		// swap the descriptorSets, which each allow access to the SSBOs
		// in one configuration.
		// We also want to flip what SSBO we draw with in the next
		// pass through the graphics pipeline.
		// Feel free to use std::swap here. You should need it twice.
	}

	// Record command buffers for drawing using the graphics pipeline
	void buildCommandBuffers()
	{
		// Destroy command buffers if already present
		if (!checkCommandBuffers())
		{
			destroyCommandBuffers();
			createCommandBuffers();
		}
		VkCommandBufferBeginInfo cmdBufInfo = vkTools::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vkTools::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			// Draw the particle system using the update vertex buffer

			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vkTools::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

			VkRect2D scissor = vkTools::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);

			// LOOK: we always run the graphics pipeline with compute.storageBufferB.
			// How does this influence flip-flopping in draw()?
			// Try drawing with storageBufferA instead of storageBufferB. What happens? Why?
			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindVertexBuffers(drawCmdBuffers[i], VERTEX_BUFFER_BIND_ID, 1, &compute.storageBufferB.buffer, offsets);
			vkCmdDraw(drawCmdBuffers[i], PARTICLE_COUNT, 1, 0, 0);

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}

	}

	// LOOK: Record a command buffer for compute using the compute pipeline
	void buildComputeCommandBuffer()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vkTools::initializers::commandBufferBeginInfo();

		// LOOK: we only have a single commandBuffer for compute, so we have to wait until
		// it is done executing before we can overwrite it.
		// In practice, we may want to have two command buffers for compute so we can
		// fill one without waiting for the other to finish.
		// This also makes the vkWaitForFences in draw() redundant.
		// What happens when you remove it? Why?
		vkWaitForFences(device, 1, &compute.fence, VK_TRUE, UINT64_MAX);

		// Start recording commands into the command buffer!
		VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));

		// Add memory barrier to ensure that the graphics pipeline has fetched attributes before compute starts to write to the buffer
		VkBufferMemoryBarrier bufferBarrier = vkTools::initializers::bufferMemoryBarrier();
		bufferBarrier.buffer = compute.storageBufferA.buffer;
		bufferBarrier.size = compute.storageBufferA.descriptor.range;
		bufferBarrier.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;						// Vertex shader invocations have finished reading from the buffer
		bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;								// Compute shader wants to write to the buffer
		
		// Compute and graphics queue may have different queue families (see VulkanDevice::createLogicalDevice)
		// For the barrier to work across different queues, we need to set their family indices
		bufferBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;			// Required as compute and graphics queue may have different families
		bufferBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;			// Required as compute and graphics queue may have different families

		vkCmdPipelineBarrier(
			compute.commandBuffer,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_FLAGS_NONE,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr);

		// Record a binding of the pipeline to the command.
		vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);

		// Record a binding of both of our descriptorSets to the pipeline.
		// LOOK: how does this binding influence how you should flip-flop?
		vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, compute.descriptorSets, 0, 0);

		// Record a dispatch of the compute job
		vkCmdDispatch(compute.commandBuffer, PARTICLE_COUNT / 16, 1, 1);

		// Add memory barrier to ensure that compute shader has finished writing to the buffer
		// Without this the (rendering) vertex shader may display incomplete results (partial data from last frame)
		bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;								// Compute shader has finished writes to the buffer
		bufferBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;						// Vertex shader invocations want to read from the buffer
		bufferBarrier.buffer = compute.storageBufferA.buffer;
		bufferBarrier.size = compute.storageBufferA.descriptor.range;

		// Compute and graphics queue may have different queue families (see VulkanDevice::createLogicalDevice)
		// For the barrier to work across different queues, we need to set their family indices
		bufferBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;			// Required as compute and graphics queue may have different families
		bufferBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;			// Required as compute and graphics queue may have different families

		vkCmdPipelineBarrier(
			compute.commandBuffer,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_FLAGS_NONE,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr);

		// Stop recording commands to the command buffer
		vkEndCommandBuffer(compute.commandBuffer);
	}

	// LOOK: uniform buffers are updated from render()
	void updateUniformBuffers()
	{
		compute.ubo.deltaT = frameTimer * 2.5f;
		compute.ubo.rule1Distance = RULE1DISTANCE;
		compute.ubo.rule2Distance = RULE2DISTANCE;
		compute.ubo.rule3Distance = RULE3DISTANCE;
		compute.ubo.rule1Scale = RULE1SCALE;
		compute.ubo.rule2Scale = RULE2SCALE;
		compute.ubo.rule3Scale = RULE3SCALE;
		compute.ubo.particleCount = PARTICLE_COUNT;
		memcpy(compute.uniformBuffer.mapped, &compute.ubo, sizeof(compute.ubo));

		// mousePos can be used to access mouse position on screen
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();

		if (animate)
		{
			if (animStart > 0.0f)
			{
				animStart -= frameTimer * 5.0f;
			}
			else if (animStart <= 0.0f)
			{
				timer += frameTimer * 0.04f;
				if (timer > 1.f)
					timer = 0.f;
			}
		}

		updateUniformBuffers();
	}

	void toggleAnimation()
	{
		animate = !animate;
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_A:
		case GAMEPAD_BUTTON_A:
			toggleAnimation();
			break;
		}
	}
};

VULKAN_EXAMPLE_MAIN()
