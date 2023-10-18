#ifndef TEXSYN_PCA_H
#define TEXSYN_PCA_H

#include "statistics.h"
#include "image_eigen_conversions.h"
#include "eigen/Eigen/Dense"
#include "mipmaps.h"

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
	PCA (const ImageType& input, const ImageIDType &regionIDMap, uint64_t id, bool inverseSelection = false);
	PCA (const ImageType& input, const ImageScalarType &contribution);
	
	void computePCA();
	void computeProjection(unsigned int nbComponents = 0);
	void project(ImageType &res);
	void back_project(const ImageType &input, ImageType& output) const;
	
	const MatrixType &get_eigenVectors() const;
	const VectorType &get_eigenValues() const;
	const VectorType &get_mean() const;

private:

	bool isRegionIDMapUsed() const;

	// eigenvectors
	VectorType m_eigenValues;
	MatrixType m_eigenVectors;

	MatrixType m_matrix;
	MatrixType m_covariance;
	MatrixType m_projection;
	VectorType m_mean;
	
	ImageIDType m_regionIDMap;
	uint64_t m_id;
	int m_nbTexels;
	
	ImageScalarType m_contribution;
};

template<typename T>
PCA<T>::PCA():
	m_eigenValues(),
	m_eigenVectors(),
	m_matrix(),
	m_covariance(),
	m_projection(),
	m_mean(),
	m_regionIDMap(),
	m_id(0),
	m_nbTexels(0)
{}

template<typename T>
PCA<T>::PCA(const ImageType &input) :
	m_eigenValues(),
	m_eigenVectors(),
	m_matrix(),
	m_covariance(),
	m_projection(),
	m_mean(),
	m_regionIDMap(),
	m_id(0),
	m_nbTexels(input.get_width() * input.get_height())
{
	m_matrix.resize(input.get_width() * input.get_height(), input.get_nbDimensions());
	fromImageVectorToMatrix(input, m_matrix);
}

template<typename T>
PCA<T>::PCA (const ImageType& input, const ImageIDType &regionIDMap, uint64_t id, bool inverseSelection) :
	m_eigenValues(),
	m_eigenVectors(),
	m_matrix(),
	m_covariance(),
	m_projection(),
	m_mean(),
	m_regionIDMap(regionIDMap),
	m_id(id),
	m_nbTexels(0)
{
	m_matrix.resize(input.get_width() * input.get_height(), input.get_nbDimensions());
	m_nbTexels = fromImageVectorToMatrixWithRegionID(input, m_matrix, m_regionIDMap, m_id, inverseSelection);
	m_matrix.conservativeResize(m_nbTexels, input.get_nbDimensions());
}

template<typename T>
bool PCA<T>::isRegionIDMapUsed() const
{
	return m_regionIDMap.is_initialized();
}

template<typename T>
void PCA<T>::computePCA()
{
	m_eigenValues = VectorType::Zero(m_matrix.cols());
	m_eigenVectors= MatrixType::Zero(m_matrix.cols(), m_matrix.cols());
	//Subtract the mean from each column
	m_mean = m_matrix.colwise().mean();
	MatrixType centered = m_matrix.rowwise() - m_mean.transpose();
	
	if(centered.rows() > 1) //Check that covariance is non infinite
	{
		//Compute the covariance matrix
		m_covariance = centered.adjoint() * centered / double(centered.rows() -1);
		if(!m_covariance.isZero(5e-4)) //Check that covariance contains variations
		{
			//Compute the eigenvectors and eigenvalues of the covariance matrix
			Eigen::SelfAdjointEigenSolver<MatrixType> eigenSolver(m_covariance);
			m_eigenValues = eigenSolver.eigenvalues().reverse();
			m_eigenVectors = eigenSolver.eigenvectors().reverse();
		}
	}
}

template<typename T>
void PCA<T>::computeProjection(unsigned int nbComponents)
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
	if(isRegionIDMapUsed())
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
	if(isRegionIDMapUsed())
	{
		MatrixType matrix(m_nbTexels, input.get_nbDimensions());
		fromImageVectorToMatrixWithRegionID(input, matrix, m_regionIDMap, m_id);
		MatrixType back_projection = m_eigenVectors.transpose();

		// Project the matrix back onto the original space
		MatrixType matrix_inv = matrix * back_projection;
		matrix_inv.rowwise() += m_mean.transpose();
		fromMatrixToImageVectorWithRegionID(matrix_inv, output, m_regionIDMap, m_id);
	}
	else
	{
		MatrixType matrix(input.get_width() * input.get_height(), input.get_nbDimensions());
		fromImageVectorToMatrix(input, matrix);
		MatrixType back_projection = m_eigenVectors.transpose();

		// Project the matrix back onto the original space
		MatrixType matrix_inv = matrix * back_projection;
		matrix_inv.rowwise() += m_mean.transpose();
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
