import numpy as np
import matplotlib.pyplot as plt

# NOMOR 1

# PRODUCTION DATA EACH MONTH (CAN ALSO BE TRANSLATED INTO PYTHON USING PANDAS)
# Production = pd.read_excel('AOLSCEXCEL.xlsx', header=None)
productionData = np.array([1863, 1614, 2570, 1685, 2101, 1811, 2457, 2171, 2134, 2502, 2358, 2399, 2048, 2523, 2086, 2391, 2150,
                       2340, 3129, 2277, 2964, 2997, 2747, 2862, 3405, 2677, 2749, 2755, 2963, 3161, 3623, 2768, 3141, 3439,
                       3601, 3531, 3477, 3376, 4027, 3175, 3274, 3334, 3964, 3649, 3502, 3688, 3657, 4422, 4197, 4441, 4736,
                       4521, 4485, 4644, 5036, 4876, 4789, 4544, 4975, 5211, 4880, 4933, 5079, 5339, 5232, 5520, 5714, 5260,
                       6110, 5334, 5988, 6235, 6365, 6266, 6345, 6118, 6497, 6278, 6638, 6590, 6271, 7246, 6584, 6594, 7092,
                       7326, 7409, 7976, 7959, 8012, 8195, 8008, 8313, 7791, 8368, 8933, 8756, 8613, 8705, 9098, 8769, 9544,
                       9050, 9186, 10012, 9685, 9966, 10048, 10244, 10740, 10318, 10393, 10986, 10635, 10731, 11749, 11849,
                       12123, 12274, 11666, 11960, 12629, 12915, 13051, 13387, 13309, 13732, 13162, 13644, 13808, 14101, 13992,
                       15191, 15018, 14917, 15046, 15556, 15893, 16388, 16782, 16716, 17033, 16896, 17689])

# For the Month Datas. we can use numpy arange to make an array from 1 to in this case 144 for each data because 
# There are 144 Months (144 Production Data)
monthData = np.arange(1, len(productionData) + 1)

# WE CAN ALSO USE PANDAS LIBRARY
# THE EXCEL FILE IS NEEDED TO USE PANDAS

# data_rows = datas.iloc[:2, :] = SELECT THE FIRST 2 ROWS AND SELECT ALL COLUMNS
# data_rows = data_rows.T = TRANSPOSE THE DATA SO ROW BECOMES COLUMN AND COLUMS BECOMES ROWS
# x = data_rows.iloc[:, 0].values.reshape(-1, 1) = RESHAPE TO ONLY 1 COLUMN AND ALSO CHANGED TO NUMPY ARRAY
# y = data_rows.iloc[:, 1].values = OBTAIN PRODUCTION VALUES AND CHANGE TO NUMPY ARRAY

# FINDING THE MATHEMATICAL MODEL USING POLYNOMIAL REGRESSION
# NON-LINEAR APPROACH TO FIND THE TRENDLINE

# RUMUS = y = b0 + b1 * x + b2 * x^2 + b3 * x^3 + ... + bn * x^n 

# Degree of the polynomial
degree = 3 

# CREATE VANDERMONDE MATRIX MANUAL, BISA JUGA PAKE 
# X = np.vander(monthData, degree + 1)

# X is a design matrix
X = np.zeros((len(monthData), degree + 1)) # FILL JADI ZEROS SEPANJANG MONTHDATA
for i in range(degree + 1): # LOOP SAMPE DEGREE + 1 WHICH IS 4
    X[:, i] = monthData ** (degree - i) # FIILL VANDERMONDE MATRIX


# MENCARI KOEFISIEN DENGAN RUMUS
# beta = (X.T dotProduct X)^(-1) dotProduct X.T dotProduct y

# Calculate TRANSPOSE X DIKALI X (dotProduct yang pertama)
XT_X = np.dot(X.T, X)

# Calculate X TRANSPOSE DIKALI DATA PRODUCTION (dotProduct yang terakhir atau paling kanan)
XT_y = np.dot(X.T, productionData)

# CALCULATE KOEFISIEN TREND LINE (avoiding inverse)
coefficients = np.linalg.solve(XT_X, XT_y)

# CALCULATE TRENDLINE VALUE (MULTIPLY)
trendline = np.dot(X, coefficients)

coefficients_forReal = np.array([coefficients])

# THE COEFFICIENTS WITHOUT SCIENTIFIC NOTATIONS (SAMA APABILA DI EXCEL)
# .3 DECIMAL POINTS

print(f"Polynomial equation: f(x) = {coefficients[0]:.3f}x^3 + {coefficients[1]:.3f}x^2 + {coefficients[2]:.3f}x + {coefficients[3]:.3f}")


# NOMOR 2
# CONVERT MATH MODEL TO TAYLOR SERIES
# WE OBTAINED THE COEFFICIENTS ALREADY WHICH IS [0.0039,   -0.1344,   47.2236, 1748.5067]
# Coefficients

# TAYLOR SERIES FUNCTION
def taylor_series(x):
    return coefficients[0] * x**3 + coefficients[1] * x**2 + coefficients[2] * x + coefficients[3]

# PLUG IN EACH MONTH
taylor_series_values = taylor_series(monthData)

# Plot the original production data, trendline, and Taylor series approximation
plt.plot(monthData, productionData, 'o', label='Production Data', color='blue')
plt.plot(monthData, trendline, '-', label='Non-Linear Trendline', color='red', linewidth=2)
plt.plot(monthData, taylor_series_values, '--', label='Taylor Series Approximation', color='green', linewidth=2)
plt.xlabel('Month')
plt.ylabel('Production')
plt.title('Trendline vs Taylor Series Approximation')
plt.legend()
plt.show()

# NOMOR 3
# I WILL USE NEWTON RAPHSON FOR PREDICTION (BECAUSE IT'S GENERALLY EASIER AND FASTER THAN BISECTION)

# NEWTON RAPHSON FUNCTION
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100): #f adalah fungsinya, df turunan f, x0 initial guess, tolerance maksimum toleransi, max_iter max iteration
    x = x0 # VARIABLE BUAT NAMPUNG x0
    for _ in range(max_iter): # LOOP
        if abs(f(x)) < tol: # CEK TOLERANSI APAKAH MELEBIHI ATAU NGK
            return x
        x = x - f(x) / df(x) # RUMUS NEWTON RAPHSON X - F(X) / F'(X)
    return None 

def f(x):
    return coefficients[0] * x**3 + coefficients[1] * x**2 + coefficients[2] * x + coefficients[3] - 25000 #  ORIGINAL FUNCTION MINUS 25000

def df(x):
    return 3 * coefficients[0] * x**2 + 2 * coefficients[1] * x + coefficients[2] # DERIVATIVE / TURUNAN OF THE ORIGINAL FUNCTION

# INITIAL GUESS
x0 = 150

# FINDING THE ROOT
month_to_exceed_capacity = newton_raphson(f, df, x0) # Panggil Function
print(f"The warehouse would be full by month: {month_to_exceed_capacity:.3f}") # Warehouse penuh
print(f"EGIER will need to build a new warehouse by month: {month_to_exceed_capacity - 13 :.3f}") # Kapan harus membuat Warehouse baru



