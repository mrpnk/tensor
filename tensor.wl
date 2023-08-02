(* ::Package:: *)

(* ::Input::Initialization:: *)
writeToBin[filename_, ten_] :=
	Module[{file, dims = Dimensions @ ten},
		file = OpenWrite[filename];
		WriteString[file, Length @ dims];
		WriteString[file, " ", #]& /@ dims;
		WriteString[file, " "];
		Close[file];
		file = OpenAppend[filename, BinaryFormat -> True];
		BinaryWrite[file, Flatten @ ten, "Real32"];
		Close[file];
	]


(* ::Input::Initialization:: *)
readFromBin[filename_] :=
	Module[{file, n, dims, ten},
		file = OpenRead[filename, BinaryFormat -> True];
		n = Read[file, "Number"];
		dims = Table[Read[file, "Number"], n];
		ten = BinaryReadList[file, "Real32"];
		Close[file];
		Return @ ArrayReshape[ten, dims]
	]


(*Reads one number from the file, consumes exactly one extra character*)
readNumber[file_]:=Module[{c,digits,ret},
	digits=Reap[While[(c=Read[file,Character])=!=EndOfFile && StringMatchQ[c,DigitCharacter],Sow[c]]];
	ret=If[c===EndOfFile,c,FromDigits@StringJoin@digits[[2,1]]];
	Print["returned ",ret];
	Return@ret
]

readAllFromBin[filename_] :=
	Module[{file, n, dims, ten, res},
		file = OpenRead[filename, BinaryFormat -> True];
		res={};
		While[(n=readNumber[file])=!=EndOfFile,
		(
			dims=Table[readNumber[file], n];
			ten=BinaryReadList[file, "Real32",Times@@dims];
			AppendTo[res,ArrayReshape[ten, dims]]
		)];
		Close[file];
		Return@res;
	]


(* ::Input::Initialization:: *)
getConfusionMatrix[predicted_,true_]:=Module[{classes,counts,mat},
classes=Sort@DeleteDuplicates@true;
counts=Counts[Thread[true->predicted]];
mat=Table[Lookup[counts,from->to,0],{from,classes},{to,classes}];
Return[{mat,classes}]
]
PredictionMatrixPlot[predicted_,true_,classLabels_,options___]:=Module[{},
{mat,classes}=getConfusionMatrix[predicted,true];
MatrixPlot[mat,options,
PlotLabel->"Confusion Matrix",
Epilog->MapIndexed[Text[#1,{#2[[2]],-#2[[1]]}+{0,Length@classes+1}-0.5]&,mat,{2}],
FrameLabel->{"Actual","Predicted"},
FrameTicks->{Transpose@{Range[Length@classes],classLabels[[classes]]}},
ColorFunction->"StarryNightColors"
]
]



dynamicTakeDrop[l_, p_] := 
	MapThread[l[[# ;; #2]] &, {{0} ~Join~ Most@# + 1, #} & @ Accumulate @ p]
dynamicPartition[l_,p_]:=
	Flatten[dynamicTakeDrop[#,p]&/@Partition[l,Total@p],1]
