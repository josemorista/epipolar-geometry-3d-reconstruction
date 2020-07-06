import Matrix from "ml-matrix";
import { ndMat } from "../@types";
import fs from 'fs';
import path from 'path';

// Convert Matrix instance to NODE.Js array
export const matrix2ndMat = (m: Matrix): ndMat => {
  let resp = [];
  for (let i = 0; i < m.rows; i++) {
    resp[i] = m.getRow(i);
  }
  return resp;
};

export const getRandomInt = (min: number, max: number): number => {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

export const readCameraProjectionMatrixFromFile = (filename: string) => {
  const fileLines = fs.readFileSync(path.resolve('.', 'datasets', 'temple', 'temple_par.txt'), 'utf-8').split('\n');
  let K = [] as ndMat;
  let R = [] as ndMat;

  fileLines.forEach(line => {
    const parts = line.split(' ');
    if (parts[0] === filename) {
      for (let i = 0; i < 3; i++) {
        K[i] = [];
        for (let j = 0; j < 3; j++) {
          K[i][j] = Number(parts[(i * 3 + j) + 1]);
        }
      }
      for (let i = 0; i < 3; i++) {
        R[i] = [];
        for (let j = 0; j < 3; j++) {
          R[i][j] = Number(parts[(i * 3 + j) + 10]);
        }
      }
      for (let i = 0; i < 3; i++) {
        R[i].push(Number(parts[i + 19]));
      }
    }
  });

  return matrix2ndMat((new Matrix(K)).mmul(new Matrix(R)));
};
