import path from 'path';
import { findMatches } from './findMatches';
import { Matrix, SingularValueDecomposition } from 'ml-matrix';
import * as cv from 'opencv4nodejs';

const matches = findMatches(path.resolve('.', 'datasets', 'temple', 'temple0001.png'), path.resolve('.', 'datasets', 'temple', 'temple0003.png'));

const matrix2Array = (m: Matrix) => {
  let resp = [];
  for (let i = 0; i < m.rows; i++) {
    resp[i] = m.getRow(i);
  }
  return resp;
};

const restrictSingularity = (M: Matrix) => {
  const svdM = new SingularValueDecomposition(M);
  const U = new Matrix(svdM.leftSingularVectors);
  let S = new Matrix(svdM.diagonalMatrix);
  S.set(2, 2, 0);
  const V = new Matrix(svdM.rightSingularVectors.transpose());
  return U.mmul(S).mmul(V);
};

let A: Array<Array<number>> | Matrix = [];
matches.orbMatchedPoints.forEach(({ p1: { x, y }, p2: { x: xl, y: yl } }, index) => {
  A[index] = [
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

A = new Matrix(A);

const svdA = new SingularValueDecomposition(A);
const solutionArray = svdA.rightSingularVectors.getColumn(8);

let F: Array<Array<number>> | Matrix = [];
for (let i = 0; i < 3; i++) {
  F[i] = [];
  for (let j = 0; j < 3; j++) {
    F[i][j] = solutionArray[i * 3 + j];
  }
}

F = new Matrix(F);

F = restrictSingularity(F);

console.log(F);

const cvMat = new cv.Mat(matrix2Array(F), cv.CV_32F);
const { F: cv2Mat } = cv.findFundamentalMat(matches.orbMatchedPoints.map(el => new cv.Point2(el.p1.x, el.p1.y)), matches.orbMatchedPoints.map(el => new cv.Point2(el.p2.x, el.p2.y)), cv.FM_RANSAC);

console.log(cvMat.getDataAsArray());

const img2 = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0002.png'));

const lines = cv.computeCorrespondEpilines(matches.orbMatchedPoints.map(el => new cv.Point2(el.p1.x, el.p1.y)), 1, cvMat);
const linesOpenCv = cv.computeCorrespondEpilines(matches.orbMatchedPoints.map(el => new cv.Point2(el.p1.x, el.p1.y)), 1, cv2Mat);


lines.forEach(r => {
  img2.drawLine(new cv.Point2(0, -r.z / r.y), new cv.Point2(img2.cols, -(r.z + r.x * img2.cols) / r.y), new cv.Vec3(255, 0, 0), 1);
});

linesOpenCv.forEach(r => {
  img2.drawLine(new cv.Point2(0, -r.z / r.y), new cv.Point2(img2.cols, -(r.z + r.x * img2.cols) / r.y), new cv.Vec3(0, 0, 255), 1);
});


cv.imshowWait('result', img2);
