import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import enum 
import copy


timestep = 0.1 # in seconds
verbose = False

printif = print if verbose else lambda *a, **k: None
 
class Mode(enum.Enum): 
    swarm = 1
    torus = 2
    dpp = 3
    hpp = 4

class agent:
	def __init__(self, id, position, speed, velocity_unit_vector, theta, turning_angle):
		self.id = id
		xdata = 10 * np.arange(0,1, 0.1)
		self.pos = (xdata , np.sin(xdata) + 0.1 * np.random.randn(1), 
					np.cos(xdata) + 0.1 * np.random.randn(1)) if position is None else position
		vxdata = 0.1
		vydata = np.sin(vxdata) + 0.1 * np.random.randn(1)
		vzdata = np.cos(vxdata) + 0.1 * np.random.randn(1)
		self.velocity_unit_vector = tuple((vxdata,vydata[0],vzdata[0]))
		varray = np.array(self.velocity_unit_vector)
		varray = unit_vector(varray)
		self.velocity_unit_vector = tuple(varray)

		self.speed = 1 if speed is None else speed
		self.vision = 360 if theta is None else theta
		self.turning_angle = 40 if turning_angle is None else turning_angle


					# Math vector functions #
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def eucledian_distance(r1,r2):
	return np.linalg.norm(r1-r2)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    				
    				# END #

class Model:
	def __init__(self, num_agents, mode):
		self.num_agents = num_agents
		self.agents = []
		self.agent_distance_vector_table = np.full((self.num_agents,self.num_agents,3),np.inf)
		self.agent_distance_magnitude_table = np.full((self.num_agents,self.num_agents),np.inf)
		self.agent_new_direction = np.full((self.num_agents,3),0)
		self.agent_new_direction = self.agent_new_direction.astype(np.float32, copy=False)
		self.mode =  Mode.swarm if mode is None else mode
		if(mode == Mode.swarm):
			self.zor = 2
			self.zoo = 3
			self.zoa = 7
		elif(mode == Mode.torus):
			self.zor = 0.3
			self.zoo = 0.8
			self.zoa = 15			
		elif(mode == Mode.dpp):
			self.zor = 0.2
			self.zoo = 4
			self.zoa = 10
		else:
			self.zor = 0.5
			self.zoo = 10
			self.zoa = 20
		# xdata =  10 * np.arange(0,1, 0.1)
		xdata = 10 * np.random.random(self.num_agents)
		ydata = np.sin(xdata) + 1 * np.random.randn(self.num_agents)
		zdata = np.cos(xdata) + 1 * np.random.randn(self.num_agents)
		# print(xdata,ydata,zdata)
		for i in range(num_agents):
			self.agents.append(agent(i, (xdata[i],ydata[i],zdata[i]), speed=None, velocity_unit_vector=None, 
				theta = None, turning_angle = None)) 
		# self.original = copy.deepcopy(self.agents)
	def update_agent_position(self):
		self.update_agent_distance_table()
		for i in range(self.num_agents):
			my_distance = self.agent_distance_magnitude_table[i]
			repulsion_mode = 0 if (my_distance[my_distance <= self.zor].size == 0) else 1
			repulsion_vector = my_distance[my_distance <= self.zor]
			flag_o = 0 
			flag_a = 0
			for j in range(self.num_agents):
				distance = self.agent_distance_magnitude_table[i][j]
				if(self.agents[i].id == self.agents[j].id):
					continue
				if repulsion_mode and distance > self.zor:
					continue
				elif repulsion_mode and distance <= self.zor:
					self.agent_new_direction[i] -=  self.agent_distance_vector_table[i][j]
				elif distance > self.zor and distance <= self.zoo:
					vj = np.array(self.agents[j].velocity_unit_vector)
					vj /= 2
					self.agent_new_direction[i] +=  vj
					flag_o = 1
				elif distance > self.zoo and distance <= self.zoa:
					self.agent_new_direction[i] +=  (self.agent_distance_vector_table[i][j]/2)
					flag_a = 1

			if (not repulsion_mode) and  not (flag_o==1 and flag_a==1):
				 self.agent_new_direction[i]*=2

			if np.any(self.agent_new_direction[i]):
				v = np.array(self.agents[i].velocity_unit_vector)
				angle_to_turn = angle_between(v,self.agent_new_direction[i])
				final_angle_ratio = 1 if angle_to_turn <= self.agents[i].turning_angle else (self.agents[i].turning_angle/angle_to_turn) 
				self.agent_new_direction[i] = unit_vector(self.agent_new_direction[i])

			else:
				v = np.array(self.agents[i].velocity_unit_vector)
				self.agent_new_direction[i] = v 
		self.agent_distance_vector_table = np.full((self.num_agents,self.num_agents,3),np.inf)
		self.agent_distance_magnitude_table = np.full((self.num_agents,self.num_agents),np.inf)

	def update_agent_distance_table(self):
		for i in range(self.num_agents):
			for j in range(self.num_agents):
				if(self.agents[i].id == self.agents[j].id) or (self.agent_distance_magnitude_table[i][j]!=np.inf):
					continue
				r1 = np.array(self.agents[i].pos)
				r2 = np.array(self.agents[j].pos)
				self.agent_distance_vector_table[i][j] = unit_vector(r2 - r1)
				self.agent_distance_vector_table[j][i] = unit_vector(r1 - r2)
				self.agent_distance_magnitude_table[i][j] = eucledian_distance(r1,r2)
				self.agent_distance_magnitude_table[j][i] = self.agent_distance_magnitude_table[i][j] 


def update_plot_points(num,x,y,z, point, model):
	model.update_agent_position()
	new_data = model.agent_new_direction
	for i in range(model.num_agents):
		model.agents[i].velocity_unit_vector = tuple(new_data[i])
		x[i] += new_data[i][0]*model.agents[i].speed*timestep
		y[i] += new_data[i][1]*model.agents[i].speed*timestep
		z[i] += new_data[i][2]*model.agents[i].speed*timestep
		model.agents[i].pos = (x[i],y[i],z[i])
	ax = plt.axes(projection='3d')
	ax.set_xlim(-5,5)
	ax.set_ylim(-20,20)
	ax.set_zlim(-50,50)
	point = ax.scatter(x, y, z, color='b')

	return point


Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
num_agents = 50
FLAG = 0
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_xlim(-5,5)
ax.set_ylim(-20,20)
ax.set_zlim(-50,50)
zdata =  10 * np.random.random(num_agents)
xdata = np.sin(zdata) + 5 * np.random.randn(num_agents)
ydata = np.cos(zdata) + 5 * np.random.randn(num_agents)
# print((xdata,ydata,zdata))
point = ax.scatter([], [], [], color='b')
model = Model(num_agents , Mode.swarm)
# model = Model(10 , Mode.swarm)
ani=animation.FuncAnimation(fig, update_plot_points, frames=300, fargs=(xdata,ydata,zdata,point, model))
plt.show(block=True)
# ani.save('swarm.mp4',fps = 50)



