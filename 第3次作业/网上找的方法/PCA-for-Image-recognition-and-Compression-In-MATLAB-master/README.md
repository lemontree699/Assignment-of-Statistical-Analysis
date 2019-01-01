# PCA-for-Image-recognition-and-Compression-In-MATLAB
PCA Operation
 PCA is a useful statistical technique and a way of identifying patterns in the data and expressing the data in such a way as to highlight its similarities and differences. It transforms a number of (possibly) correlated variables into a (smaller) number of uncorrelated variables called principal components. The first principal component accounts for as much of the variability in the data as possible, and each succeeding component accounts for as much of the remaining variability as possible. When all PCs are retained, PCA is commonly known as the Karhunen–Loève transform (KLT). PCA is primarily deployed for decorrelation and dimensionality reduction.
Alternatively, PCA is mathematically defined as an orthogonal linear transformation that transforms the data to a new coordinate system such that the greatest variance by any projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on. PCA is theoretically the optimum transform for given data in least square terms. PCA essentially rotates the set of points around their mean in order to align with the first few principal components. This moves as much of the variance as possible (using a linear transformation) into the first few dimensions. The values in the remaining dimensions, therefore, tend to be highly correlated and may be dropped with minimal loss of information. PCA is often used in this manner for dimensionality reduction. PCA has the distinction of being the optimal linear transformation for keeping the subspace that has largest variance. This advantage, however, comes at the price of greater computational requirement if compared, for example, to the discrete cosine transform. Nonlinear dimensionality reduction techniques tend to be more computationally demanding than PCA. PCA is performed using two methods: 1-Covariance method, 2-Singular value Decomposition (SVD). 

Following is a detailed description of PCA using the covariance method as it is the technique used in this project. The goal is to transform a given data set X of dimension M to an alternative data set Y of smaller dimension L. Equivalently; we are seeking to find the matrix Y, where Y is the Karhunen–Loève transform (KLT) of matrix X: 

1)Organize the data set
Suppose the data at hand data comprises of a set of observations of M variables, and we want to reduce the data so that each observation can be described with only L variables, L < M. Suppose further, that the data are arranged as a set of N data vectors x1…xn with each xn representing a single grouped observation of the M variables.
Write x1….xN as column vectors, each of which has M rows.
Place the column vectors into a single matrix X of dimensions M × N.

2)Calculate the empirical mean
Find the empirical mean along each dimension m = 1, ..., M.
Place the calculated mean values into an empirical mean vector u of dimensions M × 1

3)Calculate the deviations from the mean
Mean subtraction is an integral part of the solution towards finding a principal component basis that minimizes the mean square error of approximating the data. Hence we proceed by centering the data as follows:
Subtract the empirical mean vector u from each column of the data matrix X.
Store mean-subtracted data in the M × N matrix B.

4)Find the covariance matrix
Find the M × M empirical covariance matrix C from the outer product of matrix B with itself:
= *
where  is the expected value operator,  is the outer product operator, and * is the conjugate transpose operator. Note that if B consists entirely of real numbers, which is the case in many applications, the "conjugate transpose" is the same as the regular transpose.

5)Find the Eigenvectors and Eigenvalues of the Covariance matrix
We then compute the matrix V of eigenvectors which diagonalizes the covariance matrix C: V-1 C V=D where D is the diagonal matrix of eigenvalues of C. This step will typically involve the use of a computer-based algorithm for computing eigenvectors and eigenvalues. Matrix D will take the form of an M × M diagonal matrix.

6)Rearrange the eigenvectors and eigenvalues
Rearrangement of eigenvectors and eigenvalues is accomplished by sorting the columns of the eigenvector matrix V and eigenvalue matrix D in order of decreasing eigenvalue. Here care must be taken to maintain the correct pairings between the columns in each matrix.

7)Compute the cumulative energy content for each eigenvector
The eigenvalues represent the distribution of the source data's energy among each of the eigenvectors, where the eigenvectors form a basis for the data.

8)Select a subset of the eigenvectors as basis vectors
A subset of eigenvectors referring to principal components is obtained by saving the first L columns of V as the M × L matrix W.

9)Transformation
Suppose your original image was of size 64 x 64 then the mean adjusted matrix would also be 64 x 64. The covariance matrix would be 64 x 64. Every column of the covariance matrix is an eigenvector/ principal component. Now suppose we intend to retain only 32 out of the 64 eigenvectors or principal components, the resulting matrix called feature matrix would be of size 64 x 32. The final data matrix would then be of size 32 x 64. 

Final data matrix = (Feature matrix)T x Mean adjusted data transposed matrix

For PCA decoding, the following equation describes the inverse PCA process resulting in the retrieval of the original dataset. 

Original data matrix = (Feature matrix x final dat matrix) x Mean matrix

Note that the mean matrix and the feature matrix are to be transmitted to the decoder for inverse PCA process. This introduces data dependency as inverse PCA cannot be performed until the two matrices mentioned above are known to the decoder.
