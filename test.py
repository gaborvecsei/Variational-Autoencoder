from vae_model.vae import VAE

vae = VAE()
vae_json = vae.model.to_json()

with open("vae_2.json", "w") as f:
    f.write(vae_json)
