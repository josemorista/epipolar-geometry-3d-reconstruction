import fs from 'fs';
import path from 'path';
import { ndMat } from './@types';

const fileLines = fs.readFileSync(path.resolve('.', 'datasets', 'temple', 'temple_par.txt'), 'utf-8').split('\n');

const readCameraInformationFromFile = (filename: string) => {
  let K = [] as ndMat;
  let R = [] as ndMat;
  let t = [] as Array<number>;

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
        t[i] = Number(parts[i + 19]);
      }
    }
  });

  return { K, R, t };
};

