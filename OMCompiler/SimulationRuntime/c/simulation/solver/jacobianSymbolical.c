/*
 * This file is part of OpenModelica.
 *
 * Copyright (c) 1998-2016, Open Source Modelica Consortium (OSMC),
 * c/o Linköpings universitet, Department of Computer and Information Science,
 * SE-58183 Linköping, Sweden.
 *
 * All rights reserved.
 *
 * THIS PROGRAM IS PROVIDED UNDER THE TERMS OF THE BSD NEW LICENSE OR THE
 * GPL VERSION 3 LICENSE OR THE OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
 * ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
 * RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
 * ACCORDING TO RECIPIENTS CHOICE.
 *
 * The OpenModelica software and the OSMC (Open Source Modelica Consortium)
 * Public License (OSMC-PL) are obtained from OSMC, either from the above
 * address, from the URLs: http://www.openmodelica.org or
 * http://www.ida.liu.se/projects/OpenModelica, and in the OpenModelica
 * distribution. GNU version 3 is obtained from:
 * http://www.gnu.org/copyleft/gpl.html. The New BSD License is obtained from:
 * http://www.opensource.org/licenses/BSD-3-Clause.
 *
 * This program is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, EXCEPT AS
 * EXPRESSLY SET FORTH IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE
 * CONDITIONS OF OSMC-PL.
 *
 */

 /*! \file jacobian_symbolical.c
 */


#ifdef USE_PARJAC
  #include <omp.h>
#endif


#include "simulation/solver/jacobianSymbolical.h"

#ifdef USE_PARJAC
void allocateThreadLocalJacobians(DATA* data, ANALYTIC_JACOBIAN** jacColumns)
{
  int maxTh = omp_get_max_threads();
  *jacColumns = (ANALYTIC_JACOBIAN*) malloc(maxTh*sizeof(ANALYTIC_JACOBIAN));
  const int index = data->callback->INDEX_JAC_A;
  ANALYTIC_JACOBIAN* jac = &(data->simulationInfo->analyticJacobians[index]);

  //MS Do we need this at any point?
  jac->sparsePattern = data->simulationInfo->analyticJacobians[data->callback->INDEX_JAC_A].sparsePattern;

  unsigned int columns = jac->sizeCols;
  unsigned int rows = jac->sizeRows;
  unsigned int sizeTmpVars = jac->sizeTmpVars;

  // Benchmarks indicate that it is beneficial to initialize and malloc the jacColumns using a parallel for loop.
  // Rationale: The thread working on the data initializes the data and thus have it in probably in cache.
  unsigned int i;
  for (i = 0; i < maxTh; ++i) {
    (*jacColumns)[i].sizeCols = columns;
    (*jacColumns)[i].sizeRows = rows;
    (*jacColumns)[i].sizeTmpVars = sizeTmpVars;
    (*jacColumns)[i].tmpVars    = (double*) calloc(sizeTmpVars, sizeof(double));
    (*jacColumns)[i].resultVars = (double*) calloc(rows, sizeof(double));
    (*jacColumns)[i].seedVars   = (double*) calloc(columns, sizeof(double));
    (*jacColumns)[i].sparsePattern = data->simulationInfo->analyticJacobians[data->callback->INDEX_JAC_A].sparsePattern;
  }
}
#endif

void genericParallelColoredSymbolicJacobianEvaluation(int rows, int columns, SPARSE_PATTERN* spp,
                                                      void* matrixA, ANALYTIC_JACOBIAN* jacColumns, DATA* data,
                                                      threadData_t* threadData,
                                                      void (*f)(int, int, int, double, void*, int))
{
#pragma omp parallel default(none) firstprivate(columns, rows) \
                                   shared(spp, matrixA, jacColumns, data, threadData, f)
{
#ifdef USE_PARJAC
//  printf("My id = %d of max threads= %d\n", omp_get_thread_num(), omp_get_num_threads());
  ANALYTIC_JACOBIAN* t_jac = &(jacColumns[omp_get_thread_num()]);
#else
  ANALYTIC_JACOBIAN* t_jac = jacColumns;
#endif

  unsigned int ii, j, l, nth, i;
#pragma omp for
  for(unsigned int i = 0; i < spp->maxColors; i++) {
//#ifdef USE_PARJAC
//    infoStreamPrint(LOG_STATS_V, 0, "Thread-ID %d, color i = %i\n", omp_get_thread_num(), i);
//#else
//    infoStreamPrint(LOG_STATS_V, 0, "Sequential, color i = %i\n", i);
//#endif
    for(unsigned int j=0; j < columns; j++) {
      if(spp->colorCols[j]-1 == i)
        t_jac->seedVars[j] = 1;
    }
    data->callback->functionJacA_column(data, threadData, t_jac, NULL);
    for(unsigned int j = 0; j < columns; j++) {
      if(t_jac->seedVars[j] == 1) {
        nth = spp->leadindex[j];
        while(nth < spp->leadindex[j+1]) {
          l = spp->index[nth];
          (*f)(l, j, nth, t_jac->resultVars[l], matrixA, rows);
          nth++;
        };
      }
    }

    for(j=0; j < columns; j++)
      t_jac->seedVars[j] = 0;
  } // for column
} // omp parallel
}

void freeAnalyticalJacobian(ANALYTIC_JACOBIAN* jacobian)
{
  free(jacobian->seedVars);
  free(jacobian->tmpVars);
  free(jacobian->resultVars);
  free(jacobian->sparsePattern->leadindex);
  free(jacobian->sparsePattern->index);
  free(jacobian->sparsePattern->colorCols);
  free(jacobian);
}

