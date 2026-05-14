def destroy_actors(actors):
    for actor in actors:
        if actor is None:
            continue

        try:
            if actor.is_alive:
                actor.destroy()
        except Exception as e:
            print(f"[WARN] Failed to destroy actor: {e}")