zip:
	cd submission_src && uvx rpzip -r src.zip src/ && cd ..
	cd submission_src && uvx rpzip -r weights.zip weights/ && cd ..
	mkdir -p _temp
	mv submission_src/src _temp/src
	mv submission_src/weights _temp/weights

unmove:
	mv _temp/src submission_src/
	mv _temp/weights submission_src/
	rm submission_src/src.zip
	rm submission_src/weights.zip

pack:
	rm -f submission_src/submission.csv; 
	rm -rf submission_src/cache;
	rm -f submission/log.txt; 
	rm -f submission/submission.csv; 
	rm -f submission/submission.zip; 
	just pack-submission; 
	just check-submission; 
	just test-submission