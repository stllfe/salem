from smol.agents.web import browser as agent


while True:
  try:
    # user = input("$> ").strip()
    # if not user or user == "/exit":
    #   break
    user = "who's david laid"
    answer = agent.run(user)
  except KeyboardInterrupt:
    break
