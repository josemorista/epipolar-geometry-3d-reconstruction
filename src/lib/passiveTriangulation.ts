import { Matrix, SingularValueDecomposition } from 'ml-matrix';
import { ndMat, IPoint } from '../@types';

export const calculateProjectionMatrix = (K: ndMat, R: ndMat, t: Array<number>) => {
  return (new Matrix(K)).mmul(new Matrix(R).mmul(new Matrix([[t[0]], [t[1]], [t[2]]])));
};

export const passiveTriangulation = (x1: IPoint, x2: IPoint, P1: ndMat, P2: ndMat): IPoint => {
  const [x, y] = x1;
  const [xl, yl] = x2;

  let A = [
    [x * P1[2][0] - P1[0][0]], [x * P1[2][1] - P1[0][1]], [x * P1[2][2] - P1[0][2]],
    [y * P1[2][0] - P1[2][0]], [y * P1[2][1] - P1[2][1]], [y * P1[2][2] - P1[2][2]],
    [xl * P2[2][0] - P2[0][0]], [xl * P2[2][1] - P2[0][1]], [xl * P2[2][2] - P2[0][2]],
    [yl * P2[2][0] - P2[2][0]], [yl * P2[2][1] - P2[2][1]], [yl * P2[2][2] - P2[2][2]],
  ];

  const svdA = new SingularValueDecomposition(new Matrix(A));

  return svdA.rightSingularVectors.getColumn(svdA.rightSingularVectors.columns - 1);
};