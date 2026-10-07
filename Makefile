.PHONY: get_subs

BUCKET = s3://drivendata-competition-sfmn/

get_subs:
	aws s3 cp $(BUCKET)/comp_subs references/submissions/comp_subs/ --recursive

# hi 