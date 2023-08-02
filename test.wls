#!/usr/bin/env wolframscript
(* ::Package:: *)

Import[NotebookDirectory[]<>"tensor.wl"]


filename=NotebookDirectory[]<>"test.bin";

serialize[filename,
	RandomInteger[{0,1},{2,3,4}]]
deserialize[filename]//Dimensions
deserializeAll[filename]//Dimensions


serializeAll[filename,
	{RandomInteger[{0,1},{2,3,4}],RandomInteger[{0,1},{5,6}]}]
deserialize[filename]//Dimensions
Dimensions/@deserializeAll[filename]




