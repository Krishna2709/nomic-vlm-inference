.PHONY: help build-api deploy-worker dev-worker

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build-api: ## Build the API Docker image
	cd api && docker build -t colnomic-embed-api .

deploy-worker: ## Deploy the Cloudflare Worker
	cd cloudflare && wrangler deploy

dev-worker: ## Start development server for Cloudflare Worker
	cd cloudflare && wrangler dev
