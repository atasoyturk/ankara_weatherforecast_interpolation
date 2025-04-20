import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pytz
import os

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

load_dotenv()
API_KEY = os.getenv("API_KEY")
LAT = 39.9179  # Çankaya enlem
LON = 32.8627  # Çankaya boylam
URL = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

try:
    response = requests.get(URL)
    response.raise_for_status()
    data = response.json()

    utc_times = [datetime.fromtimestamp(item['dt'], tz=timezone.utc) for item in data['list'][:8]]   
    local_tz = pytz.timezone('Europe/Istanbul')
    local_times = [utc_time.replace(tzinfo=pytz.utc).astimezone(local_tz) for utc_time in utc_times]

    hours = np.array([dt.hour for dt in local_times])
    temperatures = np.array([item['main']['temp'] for item in data['list'][:8]])

    x = hours
    y = temperatures

    print("Taken temperature values :")
    for h, t in zip(hours, temperatures):
        print(f"Saat {h}: {t:.2f} °C")

except requests.exceptions.RequestException as e:
    print(f"API Error: {e}")
    print("The Default temperature values are used.")

    x = np.array([0, 3, 6, 9, 12, 15, 18, 21])
    y = np.array([10.68, 9.73, 9.97, 14.52, 21.07, 15.74, 12.72, 11.80])
    hours = x
    temperatures = y
    print("Last days:")
    for h, t in zip(x, y):
        print(f"Saat {h}: {t:.2f} °C")

if not np.all(np.diff(x) > 0):
    print("Hours are not in order or are repeating. Correcting data...")
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    unique_indices = np.unique(x, return_index=True)[1]
    x = x[unique_indices]
    y = y[unique_indices]
    hours = x
    temperatures = y

f_1hour = interp1d(hours, temperatures, bounds_error=False, fill_value="extrapolate")
x_1hour = np.arange(0, 24, 1)
y_1hour = f_1hour(x_1hour)

print("\nSimulated temperatures:")
for h, t in zip(x_1hour, y_1hour):
    print(f"Hour {h}: {t:.2f} °C")

x_new = np.linspace(0, 23, 100)

y_new_lagrange = lagrange_interpolation(x, y, x_new)
y_val_lagrange = lagrange_interpolation(x, y, 14.5)

f_linear = interp1d(x_1hour, y_1hour, bounds_error=False, fill_value="extrapolate")
y_new_linear = f_linear(x_new)
y_val_linear = f_linear(14.5)

f_spline = CubicSpline(x_1hour, y_1hour)
y_new_spline = f_spline(x_new)
y_val_spline = f_spline(14.5)

f_true = interp1d(x_1hour, y_1hour, bounds_error=False, fill_value="extrapolate")
y_true = f_true(x_new)
y_val_true = f_true(14.5)

print(f"\nFor hour 14.30:")
print(f"Lagrange: {y_val_lagrange:.2f} °C")
print(f"Cubic Spline: {y_val_spline:.2f} °C")
print(f"Linear (Real values): {y_val_true:.2f} °C")

print(f"\nAbsolute Errors (14.30):")
print(f"Lagrange: {abs(y_val_lagrange - y_val_true):.2f} °C")
print(f"Cubic Spline: {abs(y_val_spline - y_val_true):.2f} °C")

plt.figure(figsize=(12, 8))
plt.plot(x_new, y_new_spline, label='Cubic Spline Interpolation', color='green', linestyle='--', linewidth=2)
plt.plot(x_new, y_new_lagrange, label='Lagrange Interpolation', color='blue', linestyle='--', linewidth=2)
plt.plot(x_new, y_true, label='Live Graphic', color='purple', linewidth=2, alpha=0.8)

plt.plot(x_1hour, y_1hour, 'o', label='Data points for 1 hour', color='red', markersize=5)
plt.plot(x, y, 's', label='Original values for 3 hours', color='black', markersize=8)
plt.plot(14.5, y_val_lagrange, '*', color='blue', markersize=15, label='Lagrange (14.30)')
plt.plot(14.5, y_val_spline, 'D', color='green', markersize=10, label='Cubic Spline (14.30)')
plt.plot(14.5, y_val_true, '^', color='purple', markersize=10, label='Real Values(14.30)')

plt.title('Çankaya, Ankara Weather Forecast', fontsize=20)
plt.xlabel('Hour', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.xticks(np.arange(0, 24, 1))
plt.yticks(np.arange(0, 40, 5))
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.figure(figsize=(12, 4))
plt.plot(x_new, np.abs(y_new_lagrange - y_true), label='Lagrange Error', color='blue', linestyle='--', alpha=0.5)
plt.plot(x_new, np.abs(y_new_spline - y_true), label='Cubic Spline Error', color='green')

plt.title('Interpolation Errors', fontsize=16)
plt.xlabel('Hour', fontsize=14)
plt.ylabel('Error (°C)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
