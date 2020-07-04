import { Matrix, SingularValueDecomposition, SVD } from 'ml-matrix';
import * as cv from 'opencv4nodejs';
import { IMatchPoint, ndMat } from '../@types';
import { matrix2ndMat } from './common';

export const eightPointAlgorithm = (matches: Array<IMatchPoint>) => {
  let EquationsMatrix: ndMat = [];
  matches.forEach(({ p1: { x, y }, p2: { x: xl, y: yl } }, index) => {
    EquationsMatrix[index] = [
      x * xl,
      x * yl,
      x,
      y * xl,
      y * yl,
      y,
      xl,
      yl,
      1
    ];
  });

  // SVD to solve equations system
  const svdEquations = new SingularValueDecomposition(new Matrix(EquationsMatrix));
  // Get the right singular vectors to create F matrix
  const solutionArray = svdEquations.rightSingularVectors.getColumn(8);
  let F: ndMat | Matrix = [];
  // Parse solution array to a 3x3 matrix
  for (let i = 0; i < 3; i++) {
    F[i] = [];
    for (let j = 0; j < 3; j++) {
      F[i][j] = solutionArray[i * 3 + j];
    }
  }

  const svdF = new SingularValueDecomposition(new Matrix(F));

  // Force rank 2 to F Matrix
  if (svdF.rank > 2) {
    const U = new Matrix(svdF.leftSingularVectors);
    let S = new Matrix(svdF.diagonalMatrix);
    S.set(2, 2, 0);
    const V = new Matrix(svdF.rightSingularVectors.transpose());
    F = matrix2ndMat(U.mmul(S).mmul(V));
  }

  return F;
};
