#include "texsyn_sampler.h"

namespace TexSyn
{

SamplerOrigin::SamplerOrigin():
	SamplerBase()
{}

void SamplerOrigin::generate(VectorType &vector, unsigned int nbPoints)
{
	vector.resize(nbPoints);
	return;
}

SamplerOrigin::Vec2 SamplerOrigin::next()
{
	return Vec2(0.0, 0.0);
}

SamplerUniform::SamplerUniform(uint64_t seed) :
	m_rand()
{
	if(seed != 0)
	{
		m_rand.set_seed(seed);
	}
}

void SamplerUniform::generate(VectorType &vector, unsigned int nbPoints)
{
	vector.resize(nbPoints);
	for(Vec2 &v : vector)
	{
		v = next();
	}
	return;
}

SamplerUniform::Vec2 SamplerUniform::next()
{
	return Vec2(m_rand.randf(), m_rand.randf());
}

SamplerPeriods::SamplerPeriods(uint64_t seed) :
	m_periods(),
	m_rand(),
	m_periodDenominator(0)
{
	if(seed != 0)
	{
		m_rand.set_seed(seed);
	}
}

void SamplerPeriods::setPeriods(const Vec2 &firstPeriod, const Vec2 &secondPeriod)
{
	m_periods[0] = firstPeriod;
	m_periods[1] = secondPeriod;
	m_periodDenominator = (m_periods[0][0] > 0.005 ? 1.0/m_periods[0][0] : 1.0)
		* (m_periods[0][1] > 0.005 ? 1.0/m_periods[0][1] : 1.0)
		* (m_periods[1][0] > 0.005 ? 1.0/m_periods[1][0] : 1.0)
		* (m_periods[1][1] > 0.005 ? 1.0/m_periods[1][1] : 1.0)-1;
}

void SamplerPeriods::generate(VectorType &vector, unsigned int nbPoints)
{
	vector.resize(nbPoints);
	for(Vec2 &v : vector)
	{
		v = next();
	}
	return;
}

SamplerPeriods::Vec2 SamplerPeriods::next()
{
	Vec2 v;
	double first = double(m_rand.randi_range(0, m_periodDenominator));
	double second = double(m_rand.randi_range(0, m_periodDenominator));
	double x = first * m_periods[0][0] + second * m_periods[1][0];
	double y = first * m_periods[0][1] + second * m_periods[1][1];
	v[0] = x-floor(x);
	v[1] = y-floor(y);
	return v;
}

SamplerImportance::SamplerImportance(const ImageScalarType &importanceFunction, uint64_t seed) :
	m_distribution2D(),
	m_importanceFunction(importanceFunction),
	m_rand()
{
	m_distribution2D.init(m_importanceFunction.get_vector().ptr(), importanceFunction.get_width(), importanceFunction.get_height());
	if(seed != 0)
	{
		m_rand.set_seed(seed);
	}
}

SamplerImportance::~SamplerImportance()
{}

void SamplerImportance::generate(VectorType &vector, unsigned int nbPoints)
{
	vector.resize(nbPoints);
	for(Vec2 &v : vector)
	{
		v = next();
	}
	return;
}

SamplerImportance::Vec2 SamplerImportance::next()
{
	Vec2 base(m_rand.randf(), m_rand.randf());
	return m_distribution2D.sampleContinuous(base);
}

const SamplerImportance::ImageScalarType &SamplerImportance::importanceFunction() const
{
	return m_importanceFunction;
}

SamplerImportance::Distribution1D::Distribution1D(const float *f, int n) : 
	m_func(f, f + n), 
	m_cdf(n + 1)
{
	m_cdf[0] = 0;
	for (int i = 1; i < n+1; ++i)
	{
		m_cdf[i] = m_cdf[i - 1] + m_func[i - 1] / n;
	}
	m_funcInt = m_cdf[n];
	if (m_funcInt == 0)
	{
		for (int i = 1; i < n+1; ++i)
		{
			m_cdf[i] = float(i) / float(n);
		}
	}
	else
	{
		for (int i = 1; i < n+1; ++i)
		{
			m_cdf[i] /= m_funcInt;
		}
	}
}
int SamplerImportance::Distribution1D::count() const
{
	return m_func.size();
}

float SamplerImportance::Distribution1D::sampleContinuous(float u, int *off) const
{
	int offset = findInterval(m_cdf.size(), [&](int index)
	{
		return m_cdf[index] <= u;
	});
	if (off != nullptr)
	{
		*off = offset;
	}
	float du = u - m_cdf[offset];
	if ((m_cdf[offset+1] - m_cdf[offset]) > 0)
	{
		du /= (m_cdf[offset+1] - m_cdf[offset]);
	}
	return (offset + du) / count();
}

void SamplerImportance::Distribution2D::init(const float *func, int width, int height)
{
	for (int v = 0; v < height; ++v)
	{
		m_pConditionalV.emplace_back(new Distribution1D(&func[v*width], width));
	}
	ScalarVectorType marginalFunc;
	for (int v = 0; v < height; ++v)
	{
		marginalFunc.push_back(m_pConditionalV[v]->m_funcInt);
	}
	m_pMarginal.reset(new Distribution1D(&marginalFunc[0], height));
}

SamplerImportance::Vec2 SamplerImportance::Distribution2D::sampleContinuous(const Vec2 &u) const
{
	int v;
	float d1 = m_pMarginal->sampleContinuous(u[1], &v);
	float d0 = m_pConditionalV[v]->sampleContinuous(u[0]);
	return Vec2(d0, d1);
}



SamplerManual::SamplerManual(uint64_t seed) :
	m_vectorsPool(),
	m_rand()
{
	if(seed != 0)
	{
		m_rand.set_seed(seed);
	}
	resetVectorsPool();
}

void SamplerManual::generate(VectorType &vector, unsigned int nbPoints)
{
	vector.resize(nbPoints);
	for(Vec2 &v : vector)
	{
		v = next();
	}
	return;
}

typename SamplerManual::Vec2 SamplerManual::next()
{
	return m_vectorsPool[m_rand.randi()%m_vectorsPool.size()];
}

void SamplerManual::setPool(const VectorType& vectorsPool)
{
	if(vectorsPool.size()>0)
		m_vectorsPool = vectorsPool;
	else
	{
		resetVectorsPool();
	}
}

void SamplerManual::resetVectorsPool()
{
	m_vectorsPool.resize(1);
	m_vectorsPool[0] = Vec2(0, 0);
}

}
