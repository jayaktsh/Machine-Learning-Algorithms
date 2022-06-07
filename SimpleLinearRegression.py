import numpy as np
import matplotlib.pyplot as plt

def est_coef(x, y):

   # number of observations/points
   n = np.size(x)   
   # mean of x and y vector 
   m_x = np.mean(x)
   m_y = np.mean(y)
   # cross deviation and deviation about x
   CD_xy =  np.sum(y * x) - n * m_y * m_x
   D_xx =  np.sum(x * x) - n * m_y * m_x
   # regression coefficients
   b_1 = CD_xy / D_xx
   b_0 = m_y - b_1 * m_x

   return (b_0, b_1)

def plot_regression_line (x, y, b):
   
   # plotting the actual points as scatter plot
   plt.scatter(x, y, color='m', marker = 'o', s = 30)
   # predicted response vector
   y_pred = b[0] + b[1]* x
   # plotting the regression line
   plt.plot(x, y_pred, color = 'g')
   # putting labels
   plt.xlabel('x')
   plt.ylabel('y')
   # function to show plot
   plt.show()

def main():
  # observations/data
  x = np.array([0,1,2,3,4,5,6,7,8,9])
  y = np.array([1,5,3,4,2,6,6,9,8,11])
  # estimating coefficients
  b = est_coef(x, y)
  print ("estimated coefficients are: \n b_0 = {} \n b_1 = {}".format(b[0], b[1]))
  #plotting regression line
  plot_regression_line(x, y, b)

if __name__ == "__main__":
  main()
