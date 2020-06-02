import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

class Vicsek:
    def __init__(self, sigma=0.1, L=15 , N=225, v_prey=0.5, v_pred=0.7, R=1, R_prey=3.5, R_pred=2.5, predator=False):
        """
        The constructor function, setting up all perameters and creating our entity list.
        """
        self.v_prey = v_prey #the constant velocity of the birds per time step
        self.v_pred = v_pred #the constant velocity of the predator per time step
        self.R_prey = R_prey #radius of view of prey (to pred)
        self.R_pred = R_pred #radius of view of predator (to pray)
        self.R = R #radius within which to search for neighbours
        self.N = N #number of individual bird entities
        self.L = L #size of the container(L*L)
        self.predator = predator #determine whether there exists a predator
        self.sigma = sigma
        
        self.statehistory = [] #every instance of the state through time
        self.VOPhistory = [] #every instance of the VOP through time
        self.state = [] #the current state
        self.VOP = 0 #the current VOP
        
        # if there exists a predator, then adds a row containing its information
        if self.predator:
            self.N += 1
            
        #creating the state
        state = np.zeros((self.N,3))
        #assignment of x-axis values and y-axis values
        state[:,:2] = np.random.uniform(0,self.L,(self.N,2))
        #assignment of angles
        state[:,2] = np.random.uniform(0,2*np.pi,self.N)
        
        self.update_state(state)
    
    
    def reset(self):
        """
        Reset the instance back to the initial state wiping all history.
        """
        #the original state
        tempstate = self.statehistory[0]
        self.statehistory = []
        self.VOPhistory = []
        self.update_state(tempstate) 
    
    def update_state(self, newstate):
        """
        Updates the state to be the passed 'newstate' and appends it along with its 
        calculated VOP to the history lists.

        Parameters
        ----------
        newstate : array-like
            The new position information of all entites.
        """
        self.statehistory.append(newstate)
        self.state = newstate
        self.VOP = self.get_VOP()
        self.VOPhistory.append(self.VOP)
              
    def get_VOP(self):
        """
        Uses equation (11) to calculate and return the VOP of the current state. 
        """
        #sums the x,y directions of each angle of each entity
        x, y = np.sum(np.cos(self.state[:,2])), np.sum(np.sin(self.state[:,2]))
        VOP = np.sqrt(x**2 + y**2) / self.N
        return VOP
    
    def get_distance_matrix(self):
        """
        Calculate and return the symmetric distance matrix which records the distance information of
        all entities to one another.
        """
        matrix = np.zeros((self.N,self.N))
        #itterates through each entity in pos, determining its direct distance from every other entity
        for i in range(self.N):
            dx, dy = abs(self.state[:,0] - self.state[i,0]), abs(self.state[:,1] - self.state[i,1])
            #the perodic boundary conditions for interacting with other entitys ”over the edge”
            dx[dx>self.L/2] -= self.L
            dy[dy>self.L/2] -= self.L
            #calculating the direct distance
            matrix[:,i] = np.sqrt(dx**2 + dy**2)
        return matrix    

    
    def timestep(self):
        """
        Uses the current location information and updates it by applying the formulas in the 
        Vicsek model (Will include a predator if it exists. If so there will be some differences 
        about the derivation of the predator's and prey's behaviours).
        """
        newstate = np.zeros((self.N,3))
        #to be used for the new angles of each entity, created including the random noise 
        direction = np.random.normal(0,self.sigma,self.N)
        #the distance matrix or all entities
        matrix = self.get_distance_matrix() 

        #updates non predator entity positions
        newstate[:,0] = (self.state[:,0] + 
                                 self.v_prey*np.cos(self.state[:,2]))%self.L
        newstate[:,1] = (self.state[:,1] +
                                 self.v_prey*np.sin(self.state[:,2]))%self.L


        #updates non predator entity angles
        for i in range(self.N-1):
            #if the ith entity can  see the predator (and there is one)
            if self.predator and matrix[i,-1] < self.R_prey:
                #the angle between the ith entity and the predator
                theta1=np.arctan2(newstate[i,1] - newstate[-1,1],
                                  newstate[i,0] - newstate[-1,0])
                #the distance of ith entity to the predator
                distance=matrix[i,-1]
                #adds the updated angle to the  direction array which includes the noise
                direction[i] += ((self.state[i,2] - theta1) * distance )/ self.R_pred + theta1
            #implementing equations (9) and (10)
            else:
                #the bird entities within radius R of the ith entity
                targets = np.where(matrix[i,:self.N-1]<self.R)[0]
                #their angles
                angle=self.state[targets,2]
                #the average direction
                direction_x, direction_y = np.sum(np.cos(angle)), np.sum(np.sin(angle))
                #adds the updated angle to the  direction array which includes the noise
                direction[i] += np.arctan2(direction_y, direction_x)

        if self.predator:
            #updates predator position
            newstate[-1,0] = (self.state[-1,0] + self.v_pred*np.cos(self.state[-1,2]))%self.L
            newstate[-1,1] = (self.state[-1,1] + self.v_pred*np.sin(self.state[-1,2]))%self.L

            #updates predator angle
            #the indicies of all entities visible to the predator
            targets = np.where(matrix[-1,:self.N-1]<self.R_pred)[0]
            if len(targets) != 0:
                #their angles
                angle_found = self.state[targets,2]
                # fly in the average direction of the prey found
                predator_x, predator_y = np.sum(np.cos(angle_found)), np.sum(np.sin(angle_found))
                direction[-1] += np.arctan2(predator_y,predator_x)

        #updates all entity angles
        newstate[:,2]=direction
        #updates the object with the newly calculated state
        self.update_state(newstate)
    
    def ntimesteps(self,n):
        """
        Using the previous state, calculate the updated state, and then the new VOP.
        
        Parameters
        ----------
        n : int
            The number of times the state is to be updated.
        """
        #updates the state on each itteration
        for i in range(n):
            self.timestep()

            
    def quiverplot(self,title,state=np.array([]),verbose=False): 
        """
        A function to display the current state via quiverplot (or a passed state).
        
        Parameters
        ----------
        title : string
            The title to be used for the plot.
        state : array-like, optional
            The state to be plotted, if no state if given will use the current state. 
        verbose : Boolean, optional
            If true will return the created plot instead of displaying, used 
            within video creation.
        """
        #if no state was passed, uses current state
        if state.size==0:
            state=self.state
            
        plt.figure()
        #adds all given entites to the plot
        plt.quiver(state[:,0]%self.L,state[:,1]%self.L,
                   np.cos(state[:,2]),np.sin(state[:,2]), 
                   np.arange(self.N), scale=25, width=.004)
        #adds the predator to the plot, with a larger size so it is distinguishable from other entites
        if self.predator:
            plt.quiver(state[-1][0]%self.L, state[-1][1]%self.L,
                       np.cos(state[-1][2]),np.sin(state[-1][2]), 
                       scale=12, width=.012)
        plt.xlim(0,self.L)
        plt.ylim(0,self.L)
        plt.title(title)
        #used in movie creation
        if verbose:
            return plt     
        else:
            plt.show()


    def makevideo(self,filepath,fps,title=''):
        """
        Creates a video of the changing plots of entites from each state within
        the statehistory list. 

        Parameters
        ----------
        filepath : string
            The filepath to the desired directory where the images and video will be saved.
        fps : int
            The frame rate of the created video (per second).
        """
        #if no title is given, sets a default title
        if title=='':
            title = 'sigma = {}'.format(self.sigma)
        
        
        n = len(self.statehistory)
        #imgs stores the data of each image 
        imgs = []
        #creates the plot of each state in statehistory
        for i in range(n):
            imgpath = filepath+'{}.png'.format(i)
            fig = self.quiverplot('{} seconds'.format(i//fps),self.statehistory[i],True)
            fig.savefig(imgpath)  #save the figure to file
            plt.close() 
            #gets the data of the image
            img = cv2.imread(imgpath)
            imgs.append(img)
            #deletes the now useless image
            os.remove(imgpath)
        
        #Determins the dimensions of the video from the last selected image (as all have the same dimensions)
        height, width, layers = img.shape
        size = (width,height)

        #Creates the blank video
        out = cv2.VideoWriter(filepath+title+'.avi'.format(self.sigma),
                              cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        #Adds each image to the video
        for img in imgs:
            out.write(img)
        #Saves it to the given filepath
        out.release()


    def plotvopovertime(self):
        """
        Plots the mean and stard deviation of the VOP over time from the VOPhistory list.
        """
        n = len(self.VOPhistory)
        plt.figure()
        plt.plot(np.arange(n),self.VOPhistory,label='the Vicsek order parameter')
        plt.plot([0,n],[1,1],'--')
        plt.ylim(0,1.1)
        plt.xlabel('time steps')
        plt.ylabel('the Vicsek order parameter')
        plt.title('a measure of the alignment between birds with sigma = {}'.format(self.sigma))
        plt.legend()
        plt.show()
    
    def createVOPCSV(self,filepath):
        #creates the columns and indexing
        VOPcolumn = ['sigma = {}'.format(self.sigma)]
        VOPindex = ['step = {}'.format(i) for i in range(len(self.VOPhistory))]

        #inputs the data into the column
        df = pd.DataFrame(np.array(self.VOPhistory).T, columns=VOPcolumn, index=VOPindex)
        #saves the data to a csv file at the given filepath
        df.to_csv(filepath)
    
def readCSV(filepath):
    #reads the csvfile
    df = pd.read_csv(filepath,
         delimiter=",", encoding = "ISO-8859-1")
    #extracts the data from said frame
    data = df[df.columns[1]]
    return data
        
def VOPcsv(model,n,folderpath,sigmas):
    """
    Creates csv files recording the VOP over time for each of the given sigma values.
    
    Parameters
    ----------
    model : object
        The object containing all information to be tested.
    n : Int
        The number of times each object state is to be updated.
    folderpath : String
        The path to the desired directory where the csv file will be saved.
    sigmas : list (of Int)
        The standard deviations used when calculting the noise applied to each entity.
    """
    csvFilepaths=[]
    for i in range(len(sigmas)):
        #updates its sigma to be the desired sigma
        model.sigma = sigmas[i]
        csvpath = folderpath+'sigma = {}.csv'.format(model.sigma)
        #runs the desired timesteps
        model.ntimesteps(n)
        #appends the VOPhistory to our list VOPs
        model.createVOPCSV(csvpath)
        #append the filepath of the created csv file to our list
        csvFilepaths.append(csvpath)
        #reset the vicsek
        model.reset()
        
    return csvFilepaths

def VOPplotfromCSVs(filepaths,sigmas):
    """
    From given csv files in the format of that created by the function createVOPCSV, 
    calculates the mean and standard deviation of the VOP at each sigma, (discarding 
    values before each VOP stabalises).

    Parameters
    ----------
    filepath : String
        The filepath to the csv file which is to be read from.
    sigmas : array-like   
        The standard deviations used when calculting the noise applied to each entity.
    """
    #extracts the data from csvs
    datas = [readCSV(filepath) for filepath in filepaths]
    #calculates the current mean of each set of data
    Vopmeans = [np.mean(data) for data in datas]
    
    #then iterates through each set of data until the running_avg is within 0.01 of the previous mean, then discards all data points before said point
    for i in range(len(datas)):
        running_avg = np.mean(datas[i][:25])
        t=0
        while running_avg < Vopmeans[i]-0.005 or running_avg > Vopmeans[i]+0.005:
            t+=1
            running_avg = np.mean(datas[i][t:t+25])
        datas[i]=datas[i][t+12:]
        print('for sigma = {} we found t = {}'.format(sigmas[i],t))



    #calculates the mean of the VOP at each given sigma
    Vopmeans = [np.mean(data) for data in datas]
    #calculates the standard deviation of the VOP at each given sigma
    errorbars = [np.std(data) for data in datas]

    #plots the data
    fig, ax = plt.subplots()
    ax.errorbar(sigmas, Vopmeans, yerr=errorbars, fmt='-.b', label='the average of Vicsek order parameter')
    ax.set_xlabel('sigma')
    ax.set_ylabel('Vicsek order perameter')
    ax.set_title('the n(σ) phase transition plot')
    ax.legend()
    plt.show()




