import Matrix from "ml-matrix";
import { ndMat } from "../@types";

// Convert Matrix instance to NODE.Js array
export const matrix2ndMat = (m: Matrix): ndMat => {
  let resp = [];
  for (let i = 0; i < m.rows; i++) {
    resp[i] = m.getRow(i);
  }
  return resp;
};