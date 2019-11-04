import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import enum 
import copy


timestep = 1.1 # in seconds
verbose = False

printif = print if verbose else lambda *a, **k: None
G_VAL = 270
 
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

class Mode(enum.Enum): 
    shape = 1
    goal = 2

class agent:
	def __init__(self, id, position, velocity_vector, theta, turning_angle):
		self.id = id
		xdata = 10 * np.arange(0,1, 0.1)
		self.pos = (xdata , np.sin(xdata) + 0.1 * np.random.randn(1), 
					np.cos(xdata) + 0.1 * np.random.randn(1)) if position is None else position
		self.velocity_vector = tuple((0.0,0.0))
		self.vision = 360 if theta is None else theta
		self.turning_angle = 0 if turning_angle is None else turning_angle

class Model:
	def __init__(self, num_agents, mode):
		self.num_agents = num_agents
		self.agents = []
		self.agent_distance_vector_table = np.full((self.num_agents,self.num_agents,2),np.inf)
		self.agent_distance_magnitude_table = np.full((self.num_agents,self.num_agents),np.inf)
		self.agent_new_direction = np.full((self.num_agents,2),0)
		self.agent_new_direction = self.agent_new_direction.astype(np.float32, copy=False)
		self.mode =  Mode.shape if mode is None else mode
		self.powerMode = 2
		self.maxForce = 1
		self.zone_radius = 23
		self.zone_visibility = 1.5 * self.zone_radius
		if(self.mode == Mode.shape):
			self.goal = []
		else:
			self.goal = np.array([-30,-30])
		# xdata =  10 * np.arange(0,1, 0.1)
		xdata = [np.random.randint(1, 50) for iter in range(self.num_agents)]
		ydata = [np.random.randint(1, 50) for iter in range(self.num_agents)]
		print(xdata,ydata)
		for i in range(num_agents):
			self.agents.append(agent(i, (xdata[i],ydata[i]), velocity_vector=None, 
				theta = None, turning_angle = None))

	def update_agent_position(self):
		# return
		self.update_agent_distance_table()
		# print(self.agent_distance_magnitude_table)
		for i in range(self.num_agents):
			my_distance = self.agent_distance_magnitude_table[i]
			force = 0
			for j in range(self.num_agents):
				distance = self.agent_distance_magnitude_table[i][j]
				if(self.agents[i].id == self.agents[j].id):
					continue
				if distance > self.zone_visibility:
					# print(1)
					continue
				force = G_VAL / pow(distance,self.powerMode)
				if (force > self.maxForce):
					force = self.maxForce
				if(distance < self.zone_radius):
					force *= -1
				self.agent_new_direction[i] +=  (self.agent_distance_vector_table[i][j]*force*timestep)
			if(self.mode == Mode.goal):
				r1 = np.array(self.agents[i].pos)
				r2 = self.goal
				goal_distance_vector = unit_vector(r2 - r1)
				goal_distance = eucledian_distance(r1,r2)
				force = (G_VAL * 1.7) / pow(goal_distance,self.powerMode)
				if (force > self.maxForce):
					force = self.maxForce
				self.agent_new_direction[i] +=  (goal_distance_vector*force*timestep)

			if np.any(self.agent_new_direction[i]):
				v = np.array(self.agents[i].velocity_vector)
				v += self.agent_new_direction[i]
				self.agents[i].velocity_vector = tuple(v)
		self.agent_distance_vector_table = np.full((self.num_agents,self.num_agents,2),np.inf)
		self.agent_distance_magnitude_table = np.full((self.num_agents,self.num_agents),np.inf)
		self.agent_new_direction = np.full((self.num_agents,2),0.0)

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







def update_points(num,x,y, point, model):
	model.update_agent_position()
	new_data = model.agent_new_direction
	for i in range(model.num_agents):
		x[i] = model.agents[i].pos[0]
		y[i] = model.agents[i].pos[1]
		velocity = np.array(model.agents[i].velocity_vector)
		# if(i==1):
			# print(x[i],y[i], velocity[1])
		# print(velocity)
		x[i] += velocity[0]*timestep
		y[i] += velocity[1]*timestep
		model.agents[i].pos = (x[i],y[i])
		# print(model.agents[i].velocity_vector)
		model.agents[i].velocity_vector = tuple((0.0,0.0))
		# print(model.agents[i].velocity_vector)
	ax = plt.axes()
	ax.clear()
	n = 100
	ax.set_xlim(-1*n,n)
	ax.set_ylim(-1*n,n)
	point = ax.scatter(x, y, color='b')
	if(model.mode == Mode.goal):
		point = ax.scatter(model.goal[0], model.goal[1], color='r')
	return point



Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
num_agents = 6
# print(angle_between(b[0],b[1]))
FLAG = 0
fig = plt.figure(figsize=(8,8))
fig.suptitle('Squad Goal')
ax = plt.axes()
xdata =  10 * np.random.random(num_agents)
ydata = np.sin(xdata) + 5 * np.random.randn(num_agents)
point = ax.scatter([], [], color='b')
model = Model(num_agents , Mode.goal )
ani=animation.FuncAnimation(fig, update_points, frames=200, fargs=(xdata,ydata,point, model))
plt.show(block=True)
# ani.save('squad_goal.mp4',fps=40)



