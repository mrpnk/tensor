(* ::Package:: *)

(*
To use this package:
Import[NotebookDirectory[]<>"tensor.wl"]
*)


(*Reads one number from the file, consumes exactly one extra character.*)
privateReadNextNumber[file_InputStream]:=Module[{c,digits,ret},
	digits=Reap[While[(c=Read[file,Character])=!=EndOfFile
		&& StringMatchQ[c,DigitCharacter],Sow[c]]];
	ret=If[c===EndOfFile,c,FromDigits@StringJoin@digits[[2,1]]];
	
	Return@ret
]


(*Writes the tensor content to an opened file.*)
privateSerialize[filename_,ten_,append_]:=Module[{file,dims = Dimensions@ten},
	file=If[append,OpenAppend[filename],OpenWrite[filename]];
	WriteString[file, Length @ dims];
	WriteString[file, " ", #]& /@ dims;
	WriteString[file, " "];
	Close[file];
	file=OpenAppend[filename, BinaryFormat -> True];
	BinaryWrite[file, Flatten @ ten, "Real32"];
	Close[file];
	Return@filename;
]


(* ::Input::Initialization:: *)
(*Writes the tensor content to an opened file.*)
serialize[filename_,ten_]:=Module[{},
	Return@privateSerialize[filename,ten,False];
]


(*Writes all tensors to a binary file.*)
serializeAll[filename_,tensors_]:=Module[{},
	Return@MapIndexed[privateSerialize[filename,#1,#2[[1]]!=1]&,tensors];
]


(* ::Input::Initialization:: *)
(*Loads the first tensor in a binary file.*)
deserialize[filename_]:=Module[{file, n, dims, ten},
	file = OpenRead[filename, BinaryFormat -> True];
	n = Read[file, "Number"];
	dims = Table[Read[file, "Number"], n];
	ten = BinaryReadList[file, "Real32"];
	Close[file];
	Return @ ArrayReshape[ten, dims]
]


(*Returns a list of all tensors in a binary file.*)
deserializeAll[filename_]:=Module[{file, n, dims, ten, res},
	file = OpenRead[filename, BinaryFormat -> True];
	res={};
	While[(n=privateReadNextNumber[file])=!=EndOfFile,
	(
		dims=Table[privateReadNextNumber[file], n];
		ten=BinaryReadList[file, "Real32",Times@@dims];
		AppendTo[res,ArrayReshape[ten, dims]]
	)];
	Close[file];
	Return@res;
]
