.PHONY: training submission default

default: run

run: clean
	./halite -d "240 160" "python3 MyBot.py" "python3 R2D2.py" | tee halite_out.log

clean:
	rm -f replay-* halite_out.log stdout-QBot-0.log

training:
	python3 -m dqn.learn

submission:
	rm -f submission.zip
	zip -r submission.zip MyBot.py LANGUAGE dqn_model.pkl install.sh hlt/ dqn/

