# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:24:18 2020

@author: Xichen Chao
"""
import numpy as np

""" global variables """
N = 200 # number of individual birds
L = 15 # size of the container(L*L)
v0 = 0.5 # the constant velocity of the birds per time step
v_predator = 0.7 # the constant velocity of the predator per time step
R = 1 # radius within which to search for neighbours
R_prey = 3.5 # field of view of preys
R_predator = 2.5 # field of view of predator  
sigma = 0.01 # variance of noise
T = 10 # total time steps

def init(predator=False):
    """ 
    Initialize the state space. Create a N (or N+1) by 3 numpy array which 
    contains the location information of all preys (can include the predator
    if it exists). 
    
    Parameters
    ----------
    predator : Boolean, optional
        Determine whether there exists a predator. If is true, then create a
        N+1 by 3 numpy array where the last line records the location information
        of the predator.
        
    Returns
    -------
    out : ndarray
        The first, second, and third row record the x-axis values, y-axis values 
        and angles respectively.
    """
    pos = np.zeros((N,3)) 
    pos[:,:2] = np.random.uniform(0,L,(N,2))
    # assignment of x-axis values and y-axis values
    pos[:,2] = np.random.uniform(0,2*np.pi,N)
    # assignment of angles
    if predator:
    # if there exists a predator, then add a line which contains its information
        pos = np.zeros((N+1,3))
        pos[:,:2] = np.random.uniform(0,L,(N+1,2))
        pos[:,2] = np.random.uniform(0,2*np.pi,N+1)
    return pos

def distance_matrix(pos,predator=False):
    """
    Calculate the symmetric distance matrix which records all the distance
    information of birds (can include the predator if it exists).
    
    Parameters
    ----------
    pos : array-like
        The position information of all birds.
    predator : Boolean, optional
        Determine whether there exists a predator. If is true, then create a
        N+1 by N+1 numpy array where the last line (or the last row) records
        the distance information between the predator and all birds.
    
    Returns
    -------
    out : ndarray
        In details, denote the matrix as A, then A[i,j] = A[j,i], which is the
        distance between bird_i and bird_j.
    """
    if predator: 
    # if there exists a predator
        matrix = np.zeros((N+1,N+1))
    # Use the periodic boundary conditions to compute the distance between
    # two birds. Note that we use two methods to reduce the time cost:
    # 1. an equivalent description of distances in periodic boundary conditions
    # 2. vectrisations of computation
    # The details are shown in the report.
        for i in range(N+1):
            dx, dy = abs(pos[:,0] - pos[i,0]), abs(pos[:,1] - pos[i,1])
            dx[dx>L/2] -= L
            dy[dy>L/2] -= L
            matrix[:,i] = np.sqrt(dx**2 + dy**2)
        return matrix
    matrix = np.zeros((N,N))
    for i in range(N):
        dx, dy = abs(pos[:,0] - pos[i,0]), abs(pos[:,1] - pos[i,1])
        dx[dx>L/2] -= L
        dy[dy>L/2] -= L
        matrix[:,i] = np.sqrt(dx**2 + dy**2)
    return matrix

def replacement_and_VOP(state,predator=False):
    """
    Input the location information and update it by applying the formulas in
    the Vicsek model (Can include a predator if it exists. If so there will
    be some differences about the derivation of the predator's behaviour).
    
    Additionally, if there is no predator then the output should contains the
    VOP (Vicsek order parameter) as well.
    
    Parameters
    ----------
    state : array-like
        The position information of all birds.
    predator : Boolean, optional
        Determine whether there exists a predator. If is true, the predator's
        behaviour will be considered separately.
        
    Returns
    -------
    out : ndarray (can be a tuple if there is no predator and the second part
        is the VOP)
        The updated location information of all birds.
    """
    if predator == False:
        direction = np.random.normal(0,sigma,N)
    # add the noise which has the distribution N(0,sigma^2)
        matrix = distance_matrix(state)
        state[:,0] = (state[:,0] + v0*np.cos(state[:,2]))%L 
        state[:,1] = (state[:,1] + v0*np.sin(state[:,2]))%L
    # Update the location information by using current positions and angles.
    # Mode L to maintain that all birds appear in the container.
        for i in range(N):
            target = np.where(matrix[i,:]<R)[0]
    # the birds which have influences to bird_i
            angle = state[target,2]
            direction_x, direction_y = np.sum(np.cos(angle)), np.sum(np.sin(angle))
            direction[i] += np.arctan2(direction_y, direction_x) 
    # the updated angle of bird_i
        state[:,2] = direction  
        x, y = np.sum(np.cos(direction)), np.sum(np.sin(direction))
        VOP = np.sqrt(x**2 + y**2) / N
    # compute the Vicsek order parameter
        return state, VOP
    else:
    # if there exists a predator
        direction = np.random.normal(0,sigma,N+1)
        matrix = distance_matrix(state,predator=True)
        state[:N,0] = (state[:N,0] + v0*np.cos(state[:N,2]))%L
        state[:N,1] = (state[:N,1] + v0*np.sin(state[:N,2]))%L
        state[N,0] = (state[N,0] + v_predator*np.cos(state[N,2]))%L
        state[N,1] = (state[N,1] + v_predator*np.sin(state[N,2]))%L
    # deal with the predator separately   
        preys_found = np.where(matrix[N,:]<R_predator)[0]
    # preys which can be found by the predator
        angle_found = state[preys_found,2]
        predator_x, predator_y = np.sum(np.cos(angle_found)), np.sum(np.sin(angle_found))
    # fly in the average direction of the prey found
        direction[N] += np.arctan2(predator_y,predator_x)
    # the updated angle of the predator
        state[N,2] = direction[N] 
        for i in range(N):
            if matrix[i,N] < R_prey:
                direction[i] = state[N,2] + np.pi
    # if bird_i notice the predator it will escape directly
            else:    
                target = np.where(matrix[i,:N]<R)[0]
                angle = state[target,2]
                direction_x, direction_y = np.sum(np.cos(angle)), np.sum(np.sin(angle))
                direction[i] += np.arctan2(direction_y, direction_x)
        state[:N,2] = direction[:N]
        return state