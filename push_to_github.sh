#!/bin/bash

# 🚀 Push to GitHub Script for ICLR-Med-Compression
# 
# This script helps push the feature/20251019-mvp1 branch to GitHub
# 
# Usage:
#   1. Set your GitHub credentials first:
#      git config --global user.name "justinfungi"
#      git config --global user.email "fhj1371201288@gmail.com"
#   
#   2. For HTTPS (recommended):
#      git remote set-url origin https://github.com/Justinfungi/ICLR-Med-Compression.git
#      
#   3. Or use SSH (if you have SSH keys set up):
#      git remote set-url origin git@github.com:Justinfungi/ICLR-Med-Compression.git
#
#   4. Then run this script:
#      bash push_to_github.sh

echo "🔍 Current Git Status:"
git status --short

echo ""
echo "📋 Current Branch:"
git branch --show-current

echo ""
echo "🌐 Remote Repository:"
git remote -v

echo ""
echo "📊 Commit History (last 3):"
git log --oneline -3

echo ""
echo "🚀 Ready to push? The following command will be executed:"
echo "   git push origin feature/20251019-mvp1"
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Pushing to GitHub..."
    git push origin feature/20251019-mvp1
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully pushed to GitHub!"
        echo "🔗 Create Pull Request at: https://github.com/Justinfungi/ICLR-Med-Compression/compare/feature/20251019-mvp1"
    else
        echo "❌ Push failed. Please check your credentials and network connection."
        echo ""
        echo "💡 Troubleshooting:"
        echo "   1. Check if you have push access to the repository"
        echo "   2. Verify your GitHub credentials"
        echo "   3. Try using a Personal Access Token instead of password"
        echo "   4. Check if 2FA is enabled on your GitHub account"
    fi
else
    echo "❌ Push cancelled."
fi

echo ""
echo "📝 Next Steps:"
echo "   1. Create a Pull Request on GitHub"
echo "   2. Review the changes in the web interface"
echo "   3. Merge when ready"
