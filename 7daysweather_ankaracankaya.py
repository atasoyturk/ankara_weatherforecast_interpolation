from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt
import numpy as np

def lagrange_interpolation(x, y, x_new):
        
    n = len(x)
    y_new = np.zeros_like(x_new, dtype=float)       
    
    for i in range(n):
        li = np.ones_like(x_new, dtype=float)        
        for j in range(n):
            if j != i:                               
                li *= (x_new - x[j]) / (x[i] - x[j])
        y_new += y[i] * li
    
    return y_new

x = np.array([0, 3, 6, 9, 12, 15, 18, 21])  
y = np.array([6, 5, 7, 12, 16, 18, 15, 10])  

x_new = np.linspace(0, 23, 100)
y_new_lagrange = lagrange_interpolation(x, y, x_new)  
y_val_lagrange = lagrange_interpolation(x, y, 14.5)    

f_linear = interp1d(x, y, bounds_error=False, fill_value="extrapolate")
y_new_linear = f_linear(x_new)
y_val_linear = f_linear(14.5)

f_spline = CubicSpline(x, y)
y_new_spline = f_spline(x_new)
y_val_spline = f_spline(14.5)

y_true = f_spline(x_new)                    # Kübik spline, referans eğri olarak kabul ediliyor
y_val_true = f_spline(14.5) 

print(f"For hour 14.30, Lagrange Interpolation: {y_val_lagrange:.2f} °C")
print(f"For hour 14.30, Linear Interpolation: {y_val_linear:.2f} °C")
print(f"For hour 14.30, reference value (Cubic Spline): {y_val_true:.2f} °C")
print(f"\nAbsolute errors(14.30 PM):")
print(f"Lagrange: {abs(y_val_lagrange - y_val_true):.2f} °C")
print(f"Linear: {abs(y_val_linear - y_val_true):.2f} °C")

plt.figure(figsize=(10, 8))
plt.plot(x_new, y_new_lagrange, label='Lagrange Interpolation', color='blue')
plt.plot(x_new, y_new_linear, label='Linear Interpolation', color='orange')
plt.plot(x_new, y_true, label='True Function', color='purple', linewidth=2)
plt.plot(x, y, 'o', label='Data Points', color='red')
plt.plot(14.5, y_val_lagrange, '*', color='blue', markersize=15, label='Lagrange at 14.30') #* işareti mavi yıldız olarak gösterir
plt.plot(14.5, y_val_linear, 's', color='orange', markersize=10, label='Linear at 14.30')   #s işareti turuncu kare olarak gösterir
plt.plot(14.5, y_val_spline, 'D', color='purple', markersize=10, label='Reference(Spline) at 14.30')   #D işareti mor elmas olarak gösterir
plt.title('Weather Data of Çankaya/Ankara', fontsize=20)
plt.xlabel('Hour of the Day')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(x_new, np.abs(y_new_lagrange - y_true), label='Lagrange Interpolation Error', color='blue')
plt.plot(x_new, np.abs(y_new_linear - y_true), label='Linear Interpolation Error', color='orange')
plt.title('Error of Interpolations', fontsize=16)
plt.xlabel('Hour of the Day')
plt.ylabel('Error (°C)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
   