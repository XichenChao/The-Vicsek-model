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

def init(L,N,predator=False):
    pos = np.zeros((N,3))
    pos[:,:2] = np.random.uniform(0,L,(N,2))
    pos[:,2] = np.random.uniform(0,2*np.pi,N)
    if predator:
        pos = np.zeros((N+1,3))
        pos[:,:2] = np.random.uniform(0,L,(N+1,2))
        pos[:,2] = np.random.uniform(0,2*np.pi,N+1)
    return pos

def distance_matrix(pos,predator=False):
    if predator:
        matrix = np.zeros((N+1,N+1))
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
    if predator == False:
        direction = np.random.normal(0,sigma,N)
        matrix = distance_matrix(state)
        state[:,0] = (state[:,0] + v0*np.cos(state[:,2]))%L 
        state[:,1] = (state[:,1] + v0*np.sin(state[:,2]))%L
        for i in range(N):
            target = np.where(matrix[i,:]<R)[0]
            angle = state[target,2]
            direction_x, direction_y = np.sum(np.cos(angle)), np.sum(np.sin(angle))
            direction[i] += np.arctan2(direction_y, direction_x) 
        state[:,2] = direction  
        x, y = np.sum(np.cos(direction)), np.sum(np.sin(direction))
        VOP = np.sqrt(x**2 + y**2) / N
        return state, VOP
    else:
        direction = np.random.normal(0,sigma,N+1)
        matrix = distance_matrix(state,predator=True)
        state[:N,0] = (state[:N,0] + v0*np.cos(state[:N,2]))%L
        state[:N,1] = (state[:N,1] + v0*np.sin(state[:N,2]))%L
        state[N,0] = (state[N,0] + v_predator*np.cos(state[N,2]))%L
        state[N,1] = (state[N,1] + v_predator*np.sin(state[N,2]))%L
        
        preys_found = np.where(matrix[N,:]<R_predator)[0]
        angle_found = state[preys_found,2]
        predator_x, predator_y = np.sum(np.cos(angle_found)), np.sum(np.sin(angle_found))
        state[N,2] = np.arctan2(predator_y,predator_x)
        
        for i in range(N):
            if matrix[i,N] < R_prey:
                direction[i] = state[N,2] + np.pi
            else:    
                target = np.where(matrix[i,:N]<R)[0]
                angle = state[target,2]
                direction_x, direction_y = np.sum(np.cos(angle)), np.sum(np.sin(angle))
                direction[i] += np.arctan2(direction_y, direction_x)
        state[:N,2] = direction[:N]
        return state