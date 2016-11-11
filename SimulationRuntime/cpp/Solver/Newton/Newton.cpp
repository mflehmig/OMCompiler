/** @addtogroup solverNewton
 *
 *  @{
 */

#include <Core/ModelicaDefine.h>
#include <Core/Modelica.h>

#include <Solver/Newton/FactoryExport.h>
#include <Core/Utils/extension/logger.hpp>

#include <Solver/Newton/Newton.h>

#include <Core/Math/ILapack.h>     // needed for solution of linear system with Lapack
#include <Core/Math/Constants.h>   // definitializeion of constants like uround

#ifdef USE_LOGGER
  #define LOG_VEC(algLoop, name, vec, lc, ll) LogVec(algLoop, name, vec, lc, ll)
#else
  #define LOG_VEC(algLoop, name, vec, lc, ll)
#endif

template <typename S, typename T>
static inline void LogVec(IAlgLoop* algLoop, S name, T vec[],
                          LogCategory lc, LogLevel ll) {
  if (LOGGER_IS_SET(lc, ll)) {
    std::stringstream ss;
    ss << name << " = {";
    for (int i = 0; i < algLoop->getDimReal(); i++)
      ss <<  (i > 0? ", ": "") << vec[i];
    ss << "}";
    LOGGER_WRITE(ss.str(), lc, ll);
  }
}

Newton::Newton(IAlgLoop* algLoop, INonLinSolverSettings* settings)
  : _algLoop          (algLoop)
  , _newtonSettings   ((INonLinSolverSettings*)settings)
  , _yNames           (NULL)
  , _yNominal         (NULL)
  , _yMin             (NULL)
  , _yMax             (NULL)
  , _y                (NULL)
  , _yHelp            (NULL)
  , _yTest            (NULL)
  , _fNominal         (NULL)
  , _f                (NULL)
  , _fHelp            (NULL)
  , _fTest            (NULL)
  , _iHelp            (NULL)
  , _jac              (NULL)
  , _zeroVec          (NULL)
  , _dimSys           (0)
  , _firstCall        (true)
  , _iterationStatus  (CONTINUE)
  , _lc               (algLoop->isLinear()? LC_LS: LC_NLS)
{
}

Newton::~Newton()
{
  if (_yNames)   delete []    _yNames;
  if (_yNominal) delete []    _yNominal;
  if (_yMin)     delete []    _yMin;
  if (_yMax)     delete []    _yMax;
  if (_y)        delete []    _y;
  if (_yHelp)    delete []    _yHelp;
  if (_yTest)    delete []    _yTest;
  if (_fNominal) delete []    _fNominal;
  if (_f)        delete []    _f;
  if (_fHelp)    delete []    _fHelp;
  if (_fTest)    delete []    _fTest;
  if (_iHelp)    delete []    _iHelp;
  if (_jac)      delete []    _jac;
  if (_zeroVec)  delete []   _zeroVec;
}

void Newton::initialize()
{
  _firstCall = false;

  //(Re-) initializeialization of algebraic loop
  _algLoop->initialize();

  // Dimension of the system (number of variables)
  int
    dimDouble    = _algLoop->getDimReal(),
    dimInt       = 0,
    dimBool      = 0;

  // Check system dimension
  if (dimDouble != _dimSys) {
    _dimSys = dimDouble;

    if (_dimSys > 0) {
      // initialize of vectors of unknowns and residuals
      if (_yNames)   delete []    _yNames;
      if (_yNominal) delete []    _yNominal;
      if (_yMin)     delete []    _yMin;
      if (_yMax)     delete []    _yMax;
      if (_y)        delete []    _y;
      if (_fNominal) delete []    _fNominal;
      if (_f)        delete []    _f;
      if (_yHelp)    delete []    _yHelp;
      if (_yTest)    delete []    _yTest;
      if (_fHelp)    delete []    _fHelp;
      if (_fTest)    delete []    _fTest;
      if (_iHelp)    delete []    _iHelp;
      if (_jac)      delete []    _jac;
      if (_zeroVec)  delete []    _zeroVec;

      _yNames       = new const char* [_dimSys];
      _yNominal     = new double[_dimSys];
      _yMin         = new double[_dimSys];
      _yMax         = new double[_dimSys];
      _y            = new double[_dimSys];
      _fNominal     = new double[_dimSys];
      _f            = new double[_dimSys];
      _yHelp        = new double[_dimSys];
      _yTest        = new double[_dimSys];
      _fHelp        = new double[_dimSys];
      _fTest        = new double[_dimSys];
      _iHelp        = new long int[_dimSys];
      _jac          = new double[_dimSys*_dimSys];
      _zeroVec      = new double[_dimSys];

      _algLoop->getNamesReal(_yNames);
      _algLoop->getNominalReal(_yNominal);
      _algLoop->getMinReal(_yMin);
      _algLoop->getMaxReal(_yMax);
      std::fill(_zeroVec, _zeroVec + _dimSys, 0.0);
    }
    else {
      _iterationStatus = SOLVERERROR;
    }
  }
  LOGGER_WRITE_BEGIN("Newton: eq" + to_string(_algLoop->getEquationIndex()) +
                     " initialized", _lc, LL_DEBUG);
  LOG_VEC(_algLoop, "names", _yNames, _lc, LL_DEBUG);
  LOGGER_WRITE_END(_lc, LL_DEBUG);
}

void Newton::solve()
{
  long int
    dimRHS   = 1,        // Dimension of right hand side of linear system (=b)
    info     = 0;        // Retrun-flag of Fortran code
  int
    totSteps = 0;        // Total number of steps taken
  double
    atol = _newtonSettings->getAtol(),
    rtol = _newtonSettings->getRtol();

  // If initialize() was not called yet
  if (_firstCall)
    initialize();

  // Get current values and residuals from system
  _algLoop->getReal(_y);
  if (!_algLoop->isLinearTearing()) {
    _algLoop->evaluate();
    _algLoop->getRHS(_f);
  }

  // Reset status flag
  _iterationStatus = CONTINUE;

  LOGGER_WRITE_BEGIN("Newton: eq" + to_string(_algLoop->getEquationIndex()) +
                     " at time " + to_string(_algLoop->getSimTime()) + ":",
                     _lc, LL_DEBUG);

  while (_iterationStatus == CONTINUE) {
    if (totSteps >= _newtonSettings->getNewtMax())
      throw ModelicaSimulationError(ALGLOOP_SOLVER,
        "error solving nonlinear system (iteration limit: " + to_string(totSteps) + ")");
    // Solve linear non-torn system
    if (_algLoop->isLinear() && !_algLoop->isLinearTearing()) {
      calcJacobian(_jac, _fNominal);
      for (int i = 0; i < _dimSys; i++)
        _f[i] /= _fNominal[i];
      dgesv_(&_dimSys, &dimRHS, _jac, &_dimSys, _iHelp, _f, &_dimSys, &info);
      for (int j = 0; j < _dimSys; j++)
        _y[j] = _f[j] * _yNominal[j];
      _algLoop->setReal(_y);
      if (info != 0)
        throw ModelicaSimulationError(ALGLOOP_SOLVER,
          "error solving linear system (dgesv info: " + to_string(info) + ")");
      else
        _iterationStatus = DONE;
    }

    // Solve linear torn system
    else if (_algLoop->isLinearTearing()) {
      calcFunction(_zeroVec, _f);
      calcJacobian(_jac, _fNominal);
      for (int i = 0; i < _dimSys; i++)
        _f[i] /= _fNominal[i];
      dgesv_(&_dimSys, &dimRHS, _jac, &_dimSys, _iHelp, _f, &_dimSys, &info);
      for (int j = 0; j < _dimSys; j++)
        _y[j] = -_f[j] * _yNominal[j];
      _algLoop->setReal(_y);
      _algLoop->evaluate();
      if (info != 0)
        throw ModelicaSimulationError(ALGLOOP_SOLVER,
          "error solving linear tearing system (dgesv info: " + to_string(info) + ")");
      else
        _iterationStatus = DONE;
    }

    // Newton step for non-linear system
    else {
      LOG_VEC(_algLoop, "y" + to_string(totSteps), _y, _lc, LL_DEBUG);
      LOG_VEC(_algLoop, "f" + to_string(totSteps), _f, _lc, LL_DEBUG);

      calcJacobian(_jac, _fNominal);

      // check stopping criterion
      _iterationStatus = DONE;
      for (int i = 0; i < _dimSys; ++i) {
        if (std::abs(_f[i]) > atol + rtol * _fNominal[i]) {
          _iterationStatus = CONTINUE;
          break;
        }
      }
      if (_iterationStatus == DONE)
        break;

      // Initialize line search function
      double phi = 0.0;
      for (int i = 0; i < _dimSys; i++) {
        _f[i] /= _fNominal[i];
        phi += _f[i] * _f[i];
      }

      // Solve linear system
      dgesv_(&_dimSys, &dimRHS, _jac, &_dimSys, _iHelp, _f, &_dimSys, &info);

      if (info != 0)
        throw ModelicaSimulationError(ALGLOOP_SOLVER,
          "error solving nonlinear system (iteration: " + to_string(totSteps)
          + ", dgesv info: " + to_string(info) + ")");

      // Increase counter
      ++ totSteps;

      // De-scale result
      for (int j = 0; j < _dimSys; j++)
        _f[j] *= _yNominal[j];

      // New iterate
      double lambda = 1.0; // step size
      double alpha = 1e-4; // guard for sufficient decrease
      // first find a feasible step
      while (true) {
        for (int i = 0; i < _dimSys; i++) {
          _yHelp[i] = _y[i] - lambda * _f[i];
          _yHelp[i] = std::min(_yMax[i], std::max(_yMin[i], _yHelp[i]));
        }
        // evaluate function
        try {
          calcFunction(_yHelp, _fHelp);
        }
        catch (ModelicaSimulationError& ex) {
          if (lambda < 1e-10)
            throw ex;
          // reduce step size
          lambda *= 0.5;
          continue;
        }
        break;
      }
      // check for solution, e.g. if a linear system is treated here
      _iterationStatus = DONE;
      for (int i = 0; i < _dimSys; i++) {
        if (std::abs(_fHelp[i]) > atol + rtol * _fNominal[i]) {
          _iterationStatus = CONTINUE;
          break;
        }
      }
      // second do line search with quadratic approximation of phi(lambda)
      // C.T.Kelley: Solving Nonlinear Equations with Newton's Method,
      // no 1 in Fundamentals of Algorithms, SIAM 2003. ISBN 0-89871-546-6.
      double phiHelp = 0.0;
      for (int i = 0; i < _dimSys; i++) {
        _fHelp[i] /= _fNominal[i];
        phiHelp += _fHelp[i] * _fHelp[i];
      }
      while (_iterationStatus == CONTINUE) {
        // test half step that also serves as max bound for step reduction
        double lambdaTest = 0.5*lambda;
        for (int i = 0; i < _dimSys; i++) {
          _yTest[i] = _y[i] - lambdaTest * _f[i];
          _yTest[i] = std::min(_yMax[i], std::max(_yMin[i], _yTest[i]));
        }
        calcFunction(_yTest, _fTest);
        double phiTest = 0.0;
        for (int i = 0; i < _dimSys; i++) {
          _fTest[i] /= _fNominal[i];
          phiTest += _fTest[i] * _fTest[i];
        }
        // check for sufficient decrease of phiHelp
        // and no further decrease with phiTest
        // otherwise minimize quadratic approximation of phi(lambda)
        if (phiHelp > (1.0 - alpha * lambda) * phi ||
            phiTest < (1.0 - alpha * lambda) * phiHelp) {
          long int n = 3;
          long int ipiv[3];
          double bx[] = {phi, phiTest, phiHelp};
          double A[] = {1.0, 1.0, 1.0,
                        0.0, lambdaTest, lambda,
                        0.0, lambdaTest*lambdaTest, lambda*lambda};
          dgesv_(&n, &dimRHS, A, &n, ipiv, bx, &n, &info);
          lambda = std::max(0.1*lambda, -0.5*bx[1]/bx[2]);
          if (!(lambda >= 1e-10))
            throw ModelicaSimulationError(ALGLOOP_SOLVER,
              "Can't get sufficient decrease of solution");
          if (lambda >= lambdaTest) {
            // upper bound 0.5*lambda
            lambda = lambdaTest;
            std::copy(_yTest, _yTest + _dimSys, _yHelp);
            std::copy(_fTest, _fTest + _dimSys, _fHelp);
            phiHelp = phiTest;
          }
          else {
            for (int i = 0; i < _dimSys; i++) {
              _yHelp[i] = _y[i] - lambda * _f[i];
              _yHelp[i] = std::min(_yMax[i], std::max(_yMin[i], _yHelp[i]));
            }
            calcFunction(_yHelp, _fHelp);
            phiHelp = 0.0;
            for (int i = 0; i < _dimSys; i++) {
              _fHelp[i] /= _fNominal[i];
              phiHelp += _fHelp[i] * _fHelp[i];
            }
          }
          LOGGER_WRITE("lambda = " + to_string(lambda) +
                       ", phi = " + to_string(phi) +
                       " --> " + to_string(phiHelp),
                       _lc, LL_DEBUG);
        }
        // check for sufficient decrease
        if (phiHelp <= (1.0 - alpha * lambda) * phi)
          break;
      }
      // take iterate
      std::copy(_yHelp, _yHelp + _dimSys, _y);
      for (int i = 0; i < _dimSys; i++)
        _f[i] = _fHelp[i] * _fNominal[i];
      phi = phiHelp;
    }
  } // end while

  LOG_VEC(_algLoop, "y*", _y, _lc, LL_DEBUG);
  LOGGER_WRITE_END(_lc, LL_DEBUG);
}

IAlgLoopSolver::ITERATIONSTATUS Newton::getIterationStatus()
{
  return _iterationStatus;
}

void Newton::calcFunction(const double *y, double *residual)
{
  _algLoop->setReal(y);
  _algLoop->evaluate();
  _algLoop->getRHS(residual);
}

void Newton::stepCompleted(double time)
{

}

void Newton::calcJacobian(double *jac, double *fNominal)
{
  const double *Adata = NULL;
  std::fill(fNominal, fNominal + _dimSys, 1e2 * _newtonSettings->getAtol());

  // Use analytic Jacobian if available
  try {
    const matrix_t& A = _algLoop->getSystemMatrix();
    if (A.size1() == _dimSys && A.size2() == _dimSys) {
      Adata = A.data().begin();
      std::copy(Adata, Adata + _dimSys * _dimSys, jac);
      for (int j = 0, idx = 0; j < _dimSys; j++)
        for (int i = 0; i < _dimSys; i++, idx++)
          fNominal[i] = std::max(std::abs(jac[idx]) * _yNominal[j], fNominal[i]);
    }
  }
  catch (ModelicaSimulationError& ex) {
    LOGGER_WRITE("Analytic Jacobian failed for eq" +
                 to_string(_algLoop->getEquationIndex()) + " at time " +
                 to_string(_algLoop->getSimTime()) + ": " + ex.what(),
                 _lc, LL_WARNING);
  }

  // Alternatively apply finite differences
  if (Adata == NULL) {
    for (int j = 0; j < _dimSys; j++) {
      // Reset variables for every column
      std::copy(_y, _y + _dimSys, _yHelp);
      double stepsize = 1e2 * _newtonSettings->getRtol() * _yNominal[j];

      // Finite differences
      _yHelp[j] += stepsize;

      calcFunction(_yHelp, _fHelp);

      // Build Jacobian in Fortran format
      for (int i = 0, idx = j * _dimSys; i < _dimSys; i++, idx++) {
        jac[idx] = (_fHelp[i] - _f[i]) / stepsize;
        fNominal[i] = std::max(std::abs(jac[idx]) * _yNominal[j], fNominal[i]);
      }

      _yHelp[j] -= stepsize;
    }
  }

  // Scale Jacobian
  LOG_VEC(_algLoop, "fNominal", fNominal, _lc, LL_DEBUG);
  for (int j = 0, idx = 0; j < _dimSys; j++)
    for (int i = 0; i < _dimSys; i++, idx++)
      jac[idx] *= _yNominal[j] / fNominal[i];
}

void Newton::restoreOldValues()
{
}

void Newton::restoreNewValues()
{
}

/** @} */ // end of solverNewton