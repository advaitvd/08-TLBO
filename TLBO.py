import numpy as np
import matplotlib.pyplot as plt

class TLBO:
    def __init__(self, cost_fn,n,population_size,ub=0.99,lb=0,maxIter=100,verbose=True):
        self.cost_fn=cost_fn
        self.n=n
        self.population_size=population_size
        self.ub=ub
        self.lb=lb

        self.maxIter=maxIter
        self.population,self.PFit=self.initialize_parameters()
        self.teacher_id=np.argmin(self.PFit)
        self.teacher=self.population[self.teacher_id,:].copy()
        self.teacher_PFit=self.PFit[self.teacher_id]


    def evaluate_population(self,population):
        return np.apply_along_axis(self.cost_fn,1,population)

    def initialize_parameters(self):
        population = self.lb+(self.ub-self.lb)*np.random.random(size=(self.population_size,self.n))
        PFit = self.evaluate_population(population)
        return (population,PFit)

    def run_optimizer(self,return_logs=False):
        iters=0
        cost_log = []
        while iters<self.maxIter:
            iters+=1
            TF=np.random.randint(1,3)
            
            if return_logs:
                cost_log.append(self.teacher_PFit)
                # pop_log.append(self.population.copy())

            for i in range(self.population_size):
                # Teacher phase
                step_size = self.teacher - TF*np.mean(self.population,axis=0)
                new_sol = self.population[i,:]+np.random.random(size=step_size.shape)*step_size

                new_sol[new_sol<self.lb]=self.lb
                new_sol[new_sol>self.ub]=self.ub

                new_PFit = self.cost_fn(new_sol)

                if new_PFit<self.PFit[i]:
                    self.PFit[i]=new_PFit.copy()
                    self.population[i,:]=new_sol.copy()


                #Learner phase
                p=np.random.randint(0,self.population_size)
                Xp=self.population[p,:]
                
                if self.PFit[i]<self.PFit[p]:
                    new_sol=self.population[i,:]+np.random.random(size=step_size.shape)*(self.population[i,:]-Xp)
                else:
                    new_sol=self.population[i,:]-np.random.random(size=step_size.shape)*(self.population[i,:]-Xp)
                
                new_sol[new_sol<self.lb]=self.lb
                new_sol[new_sol>self.ub]=self.ub

                new_PFit = self.cost_fn(new_sol)

                if new_PFit<self.PFit[i]:
                    self.PFit[i]=new_PFit.copy()
                    self.population[i,:]=new_sol.copy()
                
                self.teacher_id=np.argmin(self.PFit)
                self.teacher=self.population[self.teacher_id,:].copy()
                self.teacher_PFit=self.PFit[self.teacher_id]


            if iters%10==0:
                print("Iter: {0} ----- Objective Function Value: {1}".format(iters,self.teacher_PFit))
        
        if return_logs:
            return (self.teacher,self.teacher_PFit,cost_log)
        return (self.teacher,self.teacher_PFit)



if __name__ == '__main__':
    def func(x):
        return np.sum(np.abs(x-0.5)**2)

    test = TLBO(cost_fn=func,n=5,population_size=20,ub=0.99,lb=0,maxIter=100,verbose=True)
    test.run_optimizer()
