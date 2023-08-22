#ifndef TEXSYN_PCA_H
#define TEXSYN_PCA_H

#include "texsyn_statistics.h"
#include "image_eigen.h"
#include "eigen/Eigen/Dense"
#include "mipmap_id.h"

namespace TexSyn
{

template<typename T>
class PCA
{
public :
	using DataType	= T;
	using ImageType = ImageVector<DataType>;
	using ImageScalarType = typename ImageType::ImageScalarType;
	using ImageIDType = ImageScalar<BitVector>;
	using VectorType = Eigen::VectorXd;
	using MatrixType = Eigen::MatrixXd;

	PCA();
	PCA (const ImageType& input);
	PCA (const ImageType& input, const ImageIDType &regionIDMap, uint64_t id);

	void computePCA(unsigned int nbComponents = 0);
	void project(ImageType &res);
	void back_project(const ImageType &input, ImageType& output) const;
	
	const MatrixType &get_eigenVectors() const;
	const VectorType &get_eigenValues() const;
	const VectorType &get_mean() const;

private:

	bool useRegionIDMap() const;
	void computeEigenVectors();

	// eigenvectors
	VectorType m_eigenValues;
	MatrixType m_eigenVectors;

	MatrixType m_matrix;
	MatrixType m_projection;
	VectorType m_mean;
	
	ImageIDType m_regionIDMap;
	uint64_t m_id;
	int m_nbIDs;
};

template<typename T>
PCA<T>::PCA():
	m_eigenValues(),
	m_eigenVectors(),
	m_matrix(),
	m_projection(),
	m_mean(),
	m_regionIDMap(),
	m_id(0),
	m_nbIDs(0)
{}

template<typename T>
PCA<T>::PCA(const ImageType &input) :
	m_eigenValues(),
	m_eigenVectors(),
	m_matrix(),
	m_projection(),
	m_mean(),
	m_regionIDMap(),
	m_id(0),
	m_nbIDs(0)
{
	m_matrix.resize(input.get_width() * input.get_height(), input.get_nbDimensions());
	fromImageVectorToMatrix(input, m_matrix);
	computeEigenVectors();
}

template<typename T>
PCA<T>::PCA (const ImageType& input, const ImageIDType &regionIDMap, uint64_t id) :
	m_eigenValues(),
	m_eigenVectors(),
	m_matrix(),
	m_projection(),
	m_mean(),
	m_regionIDMap(regionIDMap),
	m_id(id),
	m_nbIDs(0)
{
	m_matrix.resize(input.get_width() * input.get_height(), input.get_nbDimensions());
	m_nbIDs = fromImageVectorToMatrixWithRegionID(input, m_matrix, m_regionIDMap, m_id);
	m_matrix.conservativeResize(m_nbIDs, input.get_nbDimensions());
	computeEigenVectors();
}

template<typename T>
bool PCA<T>::useRegionIDMap() const
{
	return m_regionIDMap.is_initialized();
}

template<typename T>
void PCA<T>::computeEigenVectors()
{
	// Subtract the mean from each column
	m_mean = m_matrix.colwise().mean();
	MatrixType centered = m_matrix.rowwise() - m_mean.transpose();

	// Calculate the correlation matrix
	MatrixType cov = centered.adjoint() * centered / double(centered.rows() - 1);

	// Compute the eigenvectors and eigenvalues of the covariance matrix
	Eigen::SelfAdjointEigenSolver<MatrixType> eig(cov);
	m_eigenValues = eig.eigenvalues().reverse();
	m_eigenVectors = eig.eigenvectors().rowwise().reverse();

	// Normalize the eigenvectors
	for (int i = 0; i < m_eigenVectors.cols(); i++)
	{
		double norm = m_eigenVectors.col(i).norm();
		m_eigenVectors.col(i) /= norm;
	}
}

template<typename T>
void PCA<T>::computePCA(unsigned int nbComponents)
{
	if(nbComponents == 0)
	{
		nbComponents = m_matrix.cols();
	}
	m_projection = m_matrix.rowwise() - m_mean.transpose();
	m_projection *= m_eigenVectors.leftCols(nbComponents);
}

template<typename T>
void PCA<T>::project(ImageType &res)
{
	if(useRegionIDMap())
	{
		fromMatrixToImageVectorWithRegionID(m_projection, res, m_regionIDMap, m_id);
	}
	else
	{
		fromMatrixToImageVector(m_projection, res);
	}
}

template<typename T>
void PCA<T>::back_project(const ImageType &input, ImageType &output) const
{
	if(useRegionIDMap())
	{
		MatrixType matrix(m_nbIDs, input.get_nbDimensions());
		MatrixType projection = m_eigenVectors * m_eigenValues.asDiagonal() * m_eigenVectors.transpose();

		// Project the matrix back onto the original space
		MatrixType matrix_inv = matrix * projection.transpose() + m_mean.transpose();
		fromMatrixToImageVectorWithRegionID(matrix_inv, output, m_regionIDMap, m_id);
	}
	else
	{
		MatrixType matrix(input.get_width() * input.get_height(), input.get_nbDimensions());
		MatrixType projection = m_eigenVectors * m_eigenValues.asDiagonal() * m_eigenVectors.transpose();

		// Project the matrix back onto the original space
		MatrixType matrix_inv = matrix * projection.transpose() + m_mean.transpose();
		fromMatrixToImageVector(matrix_inv, output);
	}
}

template<typename T>
const typename PCA<T>::MatrixType &PCA<T>::get_eigenVectors() const
{
	return m_eigenVectors;
}

template<typename T>
const typename PCA<T>::VectorType &PCA<T>::get_eigenValues() const
{
	return m_eigenValues;
}

template<typename T>
const typename PCA<T>::VectorType &PCA<T>::get_mean() const
{
	return m_mean;
}

}

#endif
