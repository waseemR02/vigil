
# Contributing Guidelines

## Branching Strategy

- The **main** branch is protected to ensure it is always stable and working.
- Do not push commits directly to **main**.
- For every new feature, bug fix, or improvement:
  - Create a separate branch with a descriptive title (e.g., `feature/add-login`, `bugfix/fix-header`).
  - Work on the feature in your branch.
  - Once finished, open a Pull Request (PR) to merge your branch into **main**.
  - After code review and approval, your branch will be merged into **main**.

## Development Setup

- We primarily work with `uv` for our development sync.
- To sync your working directory using `uv`, run:
  ```
  uv sync
  ```
- This command ensures your local changes are in sync with the repository.
- Please refer to the `uv` documentation if you encounter any issues or need further customization.

## Additional Notes

- Always pull the latest changes from **main** before starting your work to avoid conflicts, rebase if needed.
- Follow the existing coding standards in your PR.
- Ensure your changes are well-tested and documented.
